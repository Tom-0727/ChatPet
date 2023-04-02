import pickle
import logging
import argparse
import transformers
import torch.nn.utils.rnn as rnn_utils


from my_modules import *
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, GPT2LMHeadModel, GPT2Config


def set_args():
    """
       Set arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, required=False,
                        help='GPU Setting')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Train without GPU')

    parser.add_argument('--model_config', default='configs/config.json', type=str, required=False,
                        help='Initial model config, from hugging face')
    parser.add_argument('--pretrained_model', default='models/model_being_fine_tuned', type=str, required=False,
                        help='Path to pretrained model')
    parser.add_argument('--dictionary_path', default='library/dictionaries/chinese.txt', type=str, required=False,
                        help='Path to dictionary')

    parser.add_argument('--dataset_path', default='data/train_50w.pkl', type=str, required=False,
                        help='Path to the dataset for training')
    parser.add_argument('--val_num', type=int, default=8000,
                        help='number of validation data')
    parser.add_argument('--max_len', default=150, type=int, required=False,
                        help='Max input length while training, longer will be truncated, detail in MyDataset')

    parser.add_argument('--log_path', default='logs/train.log', type=str, required=False,
                        help='Path to log')

    parser.add_argument('--ignore_index', default=-100, type=int, required=False,
                        help='No gradient calculation on label tokens of ignore_index')

    parser.add_argument('--epochs', default=100, type=int, required=False,
                        help='Epochs to train')
    parser.add_argument('--batch_size', default=4, type=int, required=False,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help="parameter of DataLoader, used to set multi-thread, 0 is going without multi-thread")

    parser.add_argument('--lr', default=2.6e-5, type=float, required=False,
                        help='Learning rate')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False,
                        help='AdamW optimizer epsilon rate')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, required=False,
                        help='Gradient accumulation steps, used to increase batch size in another way')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, required=False)
    parser.add_argument('--patience', type=int, default=0,
                        help="for early stopping, when 0, no early stopping.")
    parser.add_argument('--warmup_steps', type=int, default=5000,
                        help='warm up steps')

    parser.add_argument('--log_step', default=1, type=int, required=False,
                        help='Log step for loss')
    parser.add_argument('--save_model_path', default='models/', type=str, required=False,
                        help='Path to save model')

    args = parser.parse_args()
    return args


def create_logger(args):
    """
    Create logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # A handler for logging to a file
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # A handler for logging to the console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


# pads the sequences with zeros so that they all have the same length
def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels


def load_dataset(logger, args):
    """
    Load dataset (Train and Test)
    """
    logger.info("loading training dataset and validating dataset")
    dataset_path = args.dataset_path

    with open(dataset_path, "rb") as f:
        input_list = pickle.load(f)

    # Divide the dataset into training set and validation set
    val_num = args.val_num
    input_list_train = input_list[val_num:]
    input_list_val = input_list[:val_num]

    train_dataset = MyDataset(input_list_train, args.max_len)
    val_dataset = MyDataset(input_list_val, args.max_len)

    return train_dataset, val_dataset


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # return the logit with max value
    non_pad_mask = labels.ne(ignore_index)

    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return n_correct, n_word


def train_epoch(model, train_dataloader, optimizer, scheduler, logger,
                epoch, args):

    model.train()

    device = args.device
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()  # tic
    total_loss = 0  # Record the total loss of each epoch
    epoch_correct_num, epoch_total_num = 0, 0  # Record the total correct number and total number of each epoch

    # Iterate through the training set, here is one batch by one batch
    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        # capture cuda out of memory exception
        try:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

            # calculate correct number and total number in this batch
            batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
            # calculate the total correct number and total number in this epoch
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num
            # calculate the accuracy in this batch
            batch_acc = batch_correct_num / batch_total_num

            total_loss += loss.item()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            # Gradiant clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # After accumulating the gradients of several batches, update the parameters
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # parameters updating
                optimizer.step()
                # Learning rate updating
                scheduler.step()
                # clear the gradients
                optimizer.zero_grad()

            if (batch_idx + 1) % args.log_step == 0:
                logger.info(
                    "batch {} of epoch {}, loss {}, batch_acc {}, lr {}".format(
                        batch_idx + 1, epoch + 1, loss.item() * args.gradient_accumulation_steps,
                        batch_acc, scheduler.get_last_lr()))

            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    # Record the average of loss and acc of this epoch
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    logger.info(
        "epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

    # save model
    logger.info('saving model for epoch {}'.format(epoch + 1))

    model_path = os.join(args.save_model_path, 'epoch{}'.format(epoch + 1))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_to_save = model.module if hasattr(model, 'module') else model  # checks if the model is a multi-gpu model
    model_to_save.save_pretrained(model_path)

    logger.info('epoch {} finished'.format(epoch + 1))
    epoch_finish_time = datetime.now()
    logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))

    return epoch_mean_loss


def validate_epoch(model, validate_dataloader, logger, epoch, args):

    model.eval()

    logger.info("start validating")
    device = args.device
    total_loss = 0
    epoch_start_time = datetime.now()

    try:
        with torch.no_grad():
            for batch_idx, (input_ids, labels) in enumerate(validate_dataloader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids, labels=labels)
                loss = outputs.loss
                loss = loss.mean()

                total_loss += loss.item()
                del input_ids, outputs

            # Record the average of loss of this epoch
            epoch_mean_loss = total_loss / len(validate_dataloader)
            logger.info(
                "validate epoch {}: loss {}".format(epoch+1, epoch_mean_loss))
            epoch_finish_time = datetime.now()
            logger.info('time for validating one epoch: {}'.format(epoch_finish_time - epoch_start_time))
            return epoch_mean_loss
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            logger.info("WARNING: ran out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            logger.info(str(exception))
            raise exception


def train(model, logger, train_dataset, validate_dataset, args):
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn,
        drop_last=True
    )
    validate_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)

    # Early stopping if the loss on validation set does not decrease for args.patience epochs
    early_stopping = EarlyStopping(args.patience, verbose=True, save_path=args.save_model_path)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs

    # optimizer = transformers.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)

    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    logger.info('starting training')

    # Record the loss of each epoch
    train_losses, validate_losses = [], []
    # Record the smallest loss
    best_val_loss = 10000

    # Start training
    for epoch in range(args.epochs):
        # ========== train ========== #
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            logger=logger, epoch=epoch, args=args)
        train_losses.append(train_loss)

        # ========== validate ========== #
        validate_loss = validate_epoch(
            model=model, validate_dataloader=validate_dataloader,
            logger=logger, epoch=epoch, args=args)
        validate_losses.append(validate_loss)

        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            logger.info('saving current best model for epoch {}'.format(epoch + 1))
            model_path = os.join(args.save_model_path, 'min_ppl_model'.format(epoch + 1))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)

        #  If patience is 0, then do not use early stopping
        if args.patience == 0:
            continue
        early_stopping(validate_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
    logger.info('training finished')
    logger.info("train_losses:{}".format(train_losses))
    logger.info("validate_losses:{}".format(validate_losses))


def main():

    args = set_args()
    logger = create_logger(args)

    # Set the environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    args.cuda = torch.cuda.is_available() and (not args.no_cuda)
    device = 'cuda' if args.cuda else 'cpu'
    args.device = device
    logger.info('using device:{}'.format(device))

    if args.batch_size < 2048 and args.warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n')

    # tokenizer initialization
    tokenizer = BertTokenizerFast(vocab_file=args.dictionary_path,
                                  sep_token="[SEP]",
                                  pad_token="[PAD]",
                                  cls_token="[CLS]")
    args.sep_id = tokenizer.sep_token_id
    args.pad_id = tokenizer.pad_token_id
    args.cls_id = tokenizer.cls_token_id

    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    # create model
    if args.pretrained_model:  # Load pretrained model
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    else:  # Initialize Model
        model_config = GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config=model_config)
    model = model.to(device)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    assert model.config.vocab_size == tokenizer.vocab_size

    # Whether Multi-GPU Training
    if args.cuda and torch.cuda.device_count() > 1:
        model = DataParallel(model).cuda()
        # model = BalancedDataParallel(args.gpu0_bsz, model, dim=0).cuda()
        logger.info("use GPU {} to train".format(args.device))

    # Total number of parameters
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    # log the config
    logger.info("args:{}".format(args))

    # ========= Loading Dataset ========= #
    train_dataset, validate_dataset = load_dataset(logger, args)

    train(model, logger, train_dataset, validate_dataset, args)


if __name__ == '__main__':
    main()
