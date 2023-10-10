# Contextualized encoder configs

local transformer_model = "google/mt5-base";
local transformer_dim = 768; # 1024 for regular T5

local language = "ar";
local language_to_file_prefix = {
  "ar": "ara",
  "fa": "fas",
  "ko": "kor",
  "ru": "rus",
  "zh": "zho"
};

local en_data_path = "/brtx/601-nvme1/wgantt/iterx/resources/data/muc/en/tokenized/";
local tgt_data_path = "/brtx/601-nvme1/wgantt/multimuc/data/annotations/" + language + "/json";

local max_length = 1024;

# Data paths
local definition_path = "resources/data/muc/definitions.json";
local vocabulary_path = "resources/data/muc/vocabulary";

# Gold data for training and evaluation
# local train_data_path = data_path + "/tokenized-merged/train.json";
local train_data_path = {
  "en": en_data_path + "train.json",
  "tgt": tgt_data_path + "/tokenized/" + language_to_file_prefix[language] + "_train_run1.agg.filtered.json"
};
local dev_data_path = tgt_data_path + "/sf-outputs/mono_corrected/" + language + "/" + language_to_file_prefix[language] + "_dev_run1.agg.filtered.json";
local test_data_path = tgt_data_path + "/sf-outputs/mono_corrected/" + language + "/" + language_to_file_prefix[language] + "_test_run1.agg.filtered.json";

# dev and test are the same for all data configurations
local dev_gold_path = tgt_data_path + "/untokenized/" + language_to_file_prefix[language] + "_dev_run1.agg.filtered.json";
local test_gold_path = tgt_data_path + "/untokenized/" + language_to_file_prefix[language] + "_test_run1.agg.filtered.json";

# Model configs
local lexical_dropout = 0.2;
local feature_dropout = 0.3;
local iteration_cutoff = 14; # no MUC document has more than 14 templates
local span_slot_none_weight = 0.5;
local span_sampling_rate = 1.0;
# No special weight on loss for reused spans by default
local span_reuse_discount_factor = null;
local graph_encoder = {
  "type": "pytorch_transformer",
  "input_dim": transformer_dim,
  "num_layers": 3,
  "feedforward_hidden_dim": 2048,
  "num_attention_heads": 64,
  "dropout_prob": 0.3,
  "activation": "gelu"
};

local initializer = {
  "regexes": [
    [".*_span_updating_gated_sum.*weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    # Feedforwards
    [".*linear_layers.*weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    # SelfAttentiveSpanExtractor
    [".*_global_attention.*weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    # Embeddings
    ["template_embeddings.weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    ["event_type_embeddings.weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    ["event_arg_embeddings.weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    ["span_type_embeddings.weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    ["slot_type_embeddings.weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    # Graph encoder
    [".*_transformer.*linear.*\\.weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    [".*_transformer.*self_attn.*\\.in_proj_weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    [".*_transformer.*self_attn.*\\.weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    [".*_transformer.*norm.*\\.weight", {"type": "constant", "val": 1}],
    [".*_transformer.*\\.bias", {"type": "zero"}],
  ]
};

local metrics = {
  muc: {
    type: "iterx_muc",
    ignore_no_template_doc: false,
    sanitize_special_chars: true,
    doc_path: {
      [dev_data_path]: dev_gold_path
    }
  }
};

local gen_dataset_reader(is_training, skip_docs_without_spans, skip_docs_without_templates) = {
  "type": "muc",
  "definition_file": definition_path,
  "is_training": is_training,
  "skip_docs_without_spans": skip_docs_without_spans,
  "skip_docs_without_templates": skip_docs_without_templates,
  "token_indexers": {
    "tokens": {
      "type": "pretrained_transformer_mismatched",
      "model_name": transformer_model,
      "max_length": max_length
    },
  },
  # Debug options
  # "max_instances": 100,
  # "verbose": true
};

local model = {
  "type": "iterative_template_extraction",
  "definition_file": definition_path,
  # 1.0 means no discount for loss
  "grad_discount_factor": 1.0,
  "span_slot_none_weight": span_slot_none_weight,
  # > 1.0 means heavier weight on loss for reused spans
  # < 1.0 means lesser weight on loss for reused spans
  "span_reuse_discount_factor": span_reuse_discount_factor,
  "text_field_embedder": {
    "token_embedders": {
      "tokens": {
          "type": "pretrained_transformer_mismatched",
          "model_name": transformer_model,
          "max_length": max_length
      }
    }
  },
  "pairwise_scoring": true,
  "span_sampling_rate": span_sampling_rate,
  "expert_roll_out": 0.6,
  "graph_encoder": graph_encoder,
  "iteration_cutoff": iteration_cutoff,
  "lexical_dropout": lexical_dropout,
  "feature_dropout": feature_dropout,
  "initializer": initializer,
  "metrics": metrics
};

{
  # is_training: true; skip_docs_without_spans: true; skip_docs_without_templates: false
  "dataset_reader": {
    "type": "multitask",
    "readers": {
      "en": gen_dataset_reader(true, true, false), 
      "tgt": gen_dataset_reader(true, true, false)
    }
  },
  # is_training: false; skip_docs_without_spans: true; skip_docs_without_templates: true
  # NOTE: if you wish to evaluate on documents with *no* templates, this setting must be changed
  "validation_dataset_reader": gen_dataset_reader(false, true, false),
  "train_data_path": train_data_path,
  "validation_data_path": dev_data_path,
  "test_data_path": test_data_path,
  "vocabulary": {
    "type": "from_files",
    "directory": vocabulary_path,
  },
  "model": model,
  "data_loader": {
    "type": "multitask",
    "scheduler": {
        "type": "roundrobin",
        "batch_size": 1
    }
  },
  "validation_data_loader": {
    "batch_sampler": {
      "type": "bucket",
      # Explicitly specifying sorting keys since the guessing heuristic could get it wrong
      # as we are using a span field.
      "sorting_keys": ["text"],
      "batch_size": 1
    }
  },
  "trainer": {
    "num_epochs": 150, # originally 500
    "patience" : 30, # originally 50
    "num_gradient_accumulation_steps": 6,
    "validation_metric": "+iterx_muc_slot_f1",
    "grad_norm": 1.0,
    "learning_rate_scheduler": {
      "type": "polynomial_decay",
      "warmup_steps": 5000,
      "power": 2.0,
      "end_learning_rate": 1e-9
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 3e-5,
      "weight_decay": 1e-2,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}]
      ]
    },
    "cuda_device": 0
    # Uncomment if logging to Tensorboard
    # "callbacks": [
    #  {
    #    "type": "tensorboard",
    #    "should_log_parameter_statistics": true,
    #    "should_log_learning_rate": true
    #  }
    # ],
    # required if using BART as the encoder
    # "run_confidence_checks": false
  },
  # Uncomment if logging to Tensorboard
  # "callbacks": [
  # {
  #   "type": "tensorboard",
  #   "should_log_parameter_statistics": true,
  #   "should_log_learning_rate": true
  # }
  # ]
  # required if using BART as the encoder
  # "run_confidence_checks": false
}
