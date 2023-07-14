# Dependencies
import os
from datetime import datetime
import argparse
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, GPT2LMHeadModel


def set_args():
    """
    Set arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, required=False,
                        help='GPU Setting')
    parser.add_argument('--temperature', default=1, type=float, required=False,
                        help='Temperature for generation')
    parser.add_argument('--topk', default=8, type=int, required=False,
                        help='Top-k sampling')
    parser.add_argument('--topp', default=0, type=float, required=False,
                        help='Top-p sampling')
    parser.add_argument('--dictionary_path', default='library/dictionaries/chinese.txt', type=str, required=False,
                        help='Dictionary path')
    parser.add_argument('--model_path', default='models/model_test', type=str, required=False,
                        help='pre_trained model path')
    parser.add_argument('--save_record_path', default="logs/dialogues_logs/", type=str, required=False,
                        help="Path to save chat records")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="Repetition penalty parameter, increase this parameter to reduce repetition")
    parser.add_argument('--max_len', type=int, default=25,
                        help='Maximum length of each utterance, truncate if exceeds the specified length')
    parser.add_argument('--max_history_len', type=int, default=3,
                        help="Maximum length of dialogue history")
    parser.add_argument('--no_cuda', action='store_true',
                        help='Do not use GPU for prediction')
    return parser.parse_args()


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    """

    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk() return the top_k elements in the last dimension, return(values,indices)
        # ... means other dimension would automatically be deduced by computer
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # logits value of element out of topk down to -inf

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # sort logits in decline way
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def main():
    args = set_args()

    args.cuda = torch.cuda.is_available() and (not args.no_cuda)  # End up with a bool value
    device = 'cuda' if args.cuda else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # Set GPU Device you use

    # Tokenizer
    tokenizer = BertTokenizerFast(vocab_file=args.dictionary_path,
                                  sep_token="[SEP]",
                                  pad_token="[PAD]",
                                  cls_token="[CLS]")

    # Load Model
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()

    # Save chat records
    if args.save_record_path:
        if not os.path.exists(args.save_record_path):
            os.makedirs(args.save_record_path)
        records_file = open(args.save_record_path + '/Dialogue_Records.txt', 'a', encoding='utf8')  # 'a' means append
        records_file.write("----------------------------------------------------------------\n \n")
        records_file.write("Chat Records{}:\n".format(datetime.now()))

    # every utterance stored in the form of token id
    history = []
    print('Chat with Lou Lou, Quit by inputting Ctrl+D')

    # Chatting
    while True:
        try:
            text = input("You:")
            if args.save_record_path:
                records_file.write("You:{}\n".format(text))
            text_ids = tokenizer.encode(text, add_special_tokens=False)  # encode the text into token ids
            history.append(text_ids)
            input_ids = [tokenizer.cls_token_id]  # every input would start from [CLS]

            # Here we only use the last max_history_len utterances to generate the response (Important!)
            for history_id, history_utr in enumerate(history[-args.max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)
            input_ids = torch.tensor(input_ids).long().to(device)
            input_ids = input_ids.unsqueeze(0)

            response = []  # response

            # Generate tokens of max_len
            for _ in range(args.max_len):
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]

                # Add a penalty to each token generated in response to decrease repetition (important)
                for id in set(response):
                    next_token_logits[id] /= args.repetition_penalty

                # Temperature (lower temperature => more likely to sample low probability tokens)
                next_token_logits = next_token_logits / args.temperature

                # let probability of [UNK] to -inf, which means [UNK] would be produced
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')

                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)

                # torch.multinomial indicates drawing [num_samples] elements from a multinomial distribution
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token == tokenizer.sep_token_id:  # if [SEP] then response over
                    break
                response.append(next_token.item())
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)

            history.append(response)
            text = tokenizer.convert_ids_to_tokens(response)

            print("Lou Lou:" + "".join(text))
            if args.save_record_path:
                records_file.write("Lou Lou:{}\n".format("".join(text)))
        except KeyboardInterrupt:
            if args.save_record_path:
                records_file.close()
            break


if __name__ == '__main__':
    main()
