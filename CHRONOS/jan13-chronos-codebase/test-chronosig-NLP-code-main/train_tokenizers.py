

from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

import argparse
import pandas as pd


from transformers import RobertaTokenizer, RobertaTokenizerFast, BertTokenizer, BertTokenizerFast, DistilBertTokenizer


def train_byte_tokenizer(text_file, vocab_size = 32_000, min_frequency = 2,
                         special_tokens = ["<s>","<pad>", "</s>", "<unk>", "<mask>"], max_length = 512):
    """
    Prepares the tokenizer and trainer with unknown & special tokens. Byte level is typical for gpt and roberta models
    """

    print(f"Training byte level tokenizer!")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[text_file], vocab_size=vocab_size, min_frequency=min_frequency,
                    special_tokens=special_tokens)

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )

    tokenizer.enable_truncation(max_length=max_length)

    # wrap in transformer class - RobertaTokenizer for byte level
    roberta_bpe_tokenizer_fast = RobertaTokenizerFast(tokenizer_object = tokenizer, max_len=max_length)
    return roberta_bpe_tokenizer_fast

def train_wordpiece_tokenizer(text_file, vocab_size = 32000, min_frequency = 2,
                              special_tokens= ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"], max_length = 512):
    """
    Train bert word piece tokenizer...
    """
    print("Training bert word piece tokenizer")
    bert_wp_tokenizer = Tokenizer(models.WordPiece())

    # set up normalization pipeline
    bert_wp_tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])

    # pre tokenizer
    bert_wp_tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    # train
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                        special_tokens=special_tokens)

    bert_wp_tokenizer.train(files=[text_file], trainer=trainer)

    #extract cls and separator token ids
    cls_token_id = bert_wp_tokenizer.token_to_id("[CLS]")
    sep_token_id = bert_wp_tokenizer.token_to_id("[SEP]")

    # relevant post processing for bert - and next sentence prediction
    bert_wp_tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id)
        ])
    # wrap in bert tokenizer fast object
    bert_wp_tokenizer_fast = BertTokenizerFast(tokenizer_object=bert_wp_tokenizer)

    return bert_wp_tokenizer_fast


def save_tokenizer(tokenizer, save_dir, text_domain,fname):
    print(f"Saving provided tokenizer at {save_dir}/{text_domain}/{fname}")
    return tokenizer.save_pretrained(f"{save_dir}/{text_domain}/{fname}")

def tokenize(input_string, tokenizer):
    """
    Tokenizes the input string using the tokenizer provided.
    """
    output = tokenizer.encode(input_string)
    return output

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default = "F:/OxfordTempProjects/PatientTriageNLP/",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")

    parser.add_argument("--text_file",
                        default="F:/OxfordTempProjects/PatientTriageNLP/processed_data/text_MLM/all_mlm.txt",
                        type=str,
                        help="The data path to the directory containing the notes and referral data files")

    parser.add_argument("--save_dir",
                        default = "F:/OxfordTempProjects/PatientTriageNLP/nlp_development/trained_tokenizers",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")

    args = parser.parse_args()

    #train roberta fast byte tokenizer
    roberta_bp_tokenizer = train_byte_tokenizer(args.text_file)
    save_tokenizer(roberta_bp_tokenizer, args.save_dir, "eis","robertafast_byte")
    # train bert fast word piece tokenizer
    bert_wp_tokenizer = train_wordpiece_tokenizer(args.text_file)
    save_tokenizer(bert_wp_tokenizer, args.save_dir, "eis","bertfast_wordpiece")


# run script
if __name__ == "__main__":
    main()
