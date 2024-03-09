use regex::Regex;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use candle_transformers::models::quantized_t5 as t5;

use std::path::PathBuf;

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use clap::ValueEnum;
use hf_hub::{api::sync::Api, api::sync::ApiRepo, Repo, RepoType};
use tokenizers::{
    DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    Tokenizer, TokenizerImpl,
};

#[derive(Clone, Debug, Copy, ValueEnum)]
enum Which {
    T5Small,
    FlanT5Small,
    FlanT5Base,
    FlanT5Large,
    FlanT5Xl,
    FlanT5Xxl,
}

struct T5ModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: PathBuf,
}

impl T5ModelBuilder {
    pub fn load(
        model_id: Option<String>,
        revision: Option<String>,
        config: Option<String>,
        weight_file: Option<String>,
        which: Which,
    ) -> Result<(Self, Tokenizer)> {
        let device = Device::new_cuda(0)?;
        let default_model = "lmz/candle-quantized-t5".to_string();
        let (model_id, revision) = match (model_id.to_owned(), revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, "main".to_string()),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = match &config {
            Some(filename) => Self::get_local_or_remote_file(filename, &api)?,
            None => match which {
                Which::T5Small => api.get("config.json")?,
                Which::FlanT5Small => api.get("config-flan-t5-small.json")?,
                Which::FlanT5Base => api.get("config-flan-t5-base.json")?,
                Which::FlanT5Large => api.get("config-flan-t5-large.json")?,
                Which::FlanT5Xl => api.get("config-flan-t5-xl.json")?,
                Which::FlanT5Xxl => api.get("config-flan-t5-xxl.json")?,
            },
        };
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = match &weight_file {
            Some(filename) => Self::get_local_or_remote_file(filename, &api)?,
            None => match which {
                Which::T5Small => api.get("model.gguf")?,
                Which::FlanT5Small => api.get("model-flan-t5-small.gguf")?,
                Which::FlanT5Base => api.get("model-flan-t5-base.gguf")?,
                Which::FlanT5Large => api.get("model-flan-t5-large.gguf")?,
                Which::FlanT5Xl => api.get("model-flan-t5-xl.gguf")?,
                Which::FlanT5Xxl => api.get("model-flan-t5-xxl.gguf")?,
            },
        };
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: t5::Config = serde_json::from_str(&config)?;
        config.use_cache = false;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        Ok((
            Self {
                device,
                config,
                weights_filename,
            },
            tokenizer,
        ))
    }

    pub fn build_model(&self) -> Result<t5::T5ForConditionalGeneration> {
        let vb = t5::VarBuilder::from_gguf(&self.weights_filename, &self.device)?;
        Ok(t5::T5ForConditionalGeneration::load(vb, &self.config)?)
    }

    fn get_local_or_remote_file(filename: &str, api: &ApiRepo) -> Result<PathBuf> {
        let local_filename = std::path::PathBuf::from(filename);
        if local_filename.exists() {
            Ok(local_filename)
        } else {
            Ok(api.get(filename)?)
        }
    }
}
//const EN_TEST: &str = r"^[a-zA-Z0-9\s!#$%&'()*+,\-\./:;<=>?@[\]^_`{|}~①-⑽§]*$";
const EN_TEST: &str = r"^[a-zA-Z0-9\s!#$%&'()*+,\-./:;<=>?@[\\]^_`{|}~①-⑽§]*$";

fn translate(
    tokenizer: &mut TokenizerImpl<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
    builder: &T5ModelBuilder,
    model: &mut t5::T5ForConditionalGeneration,
    logits_processor: &mut LogitsProcessor,
    lang: &str,
    text: &str,
) -> Result<String> {
    let device = &builder.device;
    let mut prompt: String = "<2".to_owned();
    prompt.push_str(lang);
    prompt.push_str(">");
    prompt.push_str(text);
    let mut result = String::new();
    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let input_token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    let mut output_token_ids = [builder
        .config
        .decoder_start_token_id
        .unwrap_or(builder.config.pad_token_id) as u32]
    .to_vec();
    let encoder_output = model.encode(&input_token_ids)?;

    for index in 0.. {
        if output_token_ids.len() > 512 {
            break;
        }
        let decoder_token_ids = if index == 0 || !builder.config.use_cache {
            Tensor::new(output_token_ids.as_slice(), device)?.unsqueeze(0)?
        } else {
            let last_token = *output_token_ids.last().unwrap();
            Tensor::new(&[last_token], device)?.unsqueeze(0)?
        };
        let logits = model
            .decode(&decoder_token_ids, &encoder_output)?
            .squeeze(0)?;

        let start_at = output_token_ids.len().saturating_sub(64);
        let logits = candle_transformers::utils::apply_repeat_penalty(
            &logits,
            1.1,
            &output_token_ids[start_at..],
        )?;

        let next_token_id = logits_processor.sample(&logits)?;
        if next_token_id as usize == builder.config.eos_token_id {
            break;
        }
        output_token_ids.push(next_token_id);
        if let Some(text) = tokenizer.id_to_token(next_token_id) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            result.push_str(&text);
            std::io::stdout().flush()?;
        }
    }
    if result.is_empty() {
        result.push_str(text);
    }
    Ok(result)
}
fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 && args.len() != 5 {
        eprintln!("Usage: {} <infilename> <outfilename> <outlang>", args[0]);
        std::process::exit(1);
    }

    let infilename = &args[1];
    let outfilename = &args[2];
    let outlang = &args[3];

    let (builder, mut tokenizer) = T5ModelBuilder::load(
        Some("jbochi/madlad400-10b-mt".to_owned()),
        None,
        None,
        Some("model-q6k.gguf".to_owned()),
        Which::T5Small,
    )?;
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;

    let mut model = builder.build_model()?;
    let temperature = 0.0;

    let mut logits_processor = LogitsProcessor::new(299792458, Some(temperature), None);

    // Initialize translation model here (replace with actual initialization)
    // init('models/madlad400-10b-mt');

    let mut output_position = 0;
    if let Ok(output_file) = File::open(outfilename) {
        let mut output_file = BufReader::new(output_file);
        output_file.seek(SeekFrom::End(0))?;
        output_position = output_file.stream_position()?;
    }

    let en_regex = match Regex::new(EN_TEST) {
        Ok(regex) => regex,
        Err(err) => {
            eprintln!("Error creating regex: {}", err);
            std::process::exit(1);
        }
    };
    if args.len() == 5 {
        let translated = translate(
            tokenizer,
            &builder,
            &mut model,
            &mut logits_processor,
            outlang,
            &args[4],
        )?;
        println!("{}", translated);
        return Ok(());
    }

    let input_file = File::open(infilename)?;
    let mut input_file = BufReader::new(input_file);

    let mut output_file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(outfilename)?;

    input_file.seek(SeekFrom::Start(output_position))?;

    let mut counter = 0;
    for line in input_file.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() {
            writeln!(output_file, "")?;
            continue;
        }

        if line.starts_with('#') {
            writeln!(output_file, "{}", line)?;
        } else if line.contains('=') && (outlang != "en" || !en_regex.is_match(line)) {
            let mut parts = line.split('=');
            if let (Some(key), Some(value)) = (parts.next(), parts.next()) {
                let (prefix, value) = if value.starts_with('§') {
                    let n = value
                        .char_indices()
                        .map(|x| x.0)
                        .take(2 + 1)
                        .last()
                        .expect("empty sequence");
                    let (prefix, value) = value.split_at(n);
                    (prefix, value)
                } else {
                    ("", value)
                };
                let translated = translate(
                    tokenizer,
                    &builder,
                    &mut model,
                    &mut logits_processor,
                    outlang,
                    value,
                );
                let translated = format!("{}{}", prefix, translated?);
                println!(
                    "Translated {}={}{} to {}={}",
                    key, prefix, value, key, translated
                );
                writeln!(output_file, "{}={}", key, translated)?;
            }
        } else {
            writeln!(output_file, "{}", line)?;
        }

        counter += 1;
        if counter % 100 == 0 {
            output_file.flush()?;
        }
    }

    Ok(())
}
