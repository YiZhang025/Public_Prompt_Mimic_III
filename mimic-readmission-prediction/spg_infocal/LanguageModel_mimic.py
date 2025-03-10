import functools

print = functools.partial(print, flush=True)
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import argparse
import numpy as np
import pandas as pd

import datetime, time
from datetime import date
from Hyperparameters import args
from queue import PriorityQueue
import copy, math
from utils import *

from textdataMimic import TextDataMimic

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--modelarch', '-m')
parser.add_argument('--choose', '-c')
parser.add_argument('--use_big_emb', '-be')
parser.add_argument('--use_new_emb', '-ne')
parser.add_argument('--date', '-d')
parser.add_argument('--model_dir', '-md')
parser.add_argument('--encarch', '-ea')
cmdargs = parser.parse_args()

print("cmd args at moment: ", cmdargs)
usegpu = True

if cmdargs.gpu is None:
    usegpu = False
else:
    usegpu = True
    args['device'] = 'cuda:' + str(cmdargs.gpu)

if cmdargs.modelarch is None:
    args['model_arch'] = 'lstm'
else:
    args['model_arch'] = cmdargs.modelarch

if cmdargs.choose is None:
    args['choose'] = 0
else:
    args['choose'] = int(cmdargs.choose)

if cmdargs.use_big_emb:
    args['big_emb'] = True
else:
    args['big_emb'] = False

if cmdargs.use_new_emb:
    args['new_emb'] = True
    emb_file_path = "newemb200"
else:
    args['new_emb'] = False
    emb_file_path =  "orgembs"

if cmdargs.date is None:
    args['date'] = str(date.today())

if cmdargs.model_dir is None:
    # args['model_dir'] = "./artifacts/RCNN_IB_GAN_be_mimic3_org_embs2021-05-12.pt"
    args['model_dir'] = "./artifacts/RCNN_IB_GAN_be_mimic3_org_embs_LM2021-05-25.pt"
else:
    args["model_dir"] = str(cmdargs.model_dir)


if cmdargs.encarch is None:
    args['enc_arch'] = 'rcnn'
else:
    args['enc_arch'] = cmdargs.encarch

print(args)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), datetime.datetime.now())


class LanguageModel(nn.Module):
    def __init__(self, w2i, i2w, i2v):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(LanguageModel, self).__init__()
        print("LanguageModel creation...")

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']
        self.batch_size = args['batchSize']

        self.dtype = 'float32'

        #TODO set requires grad to true
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(i2v))
        self.embedding.weight.requires_grad = True

        if args['decunit'] == 'lstm':
            self.dec_unit = nn.LSTM(input_size=args['embeddingSize'],
                                    hidden_size=args['hiddenSize'],
                                    num_layers=args['dec_numlayer'])
        elif args['decunit'] == 'gru':
            self.dec_unit = nn.GRU(input_size=args['embeddingSize'],
                                   hidden_size=args['hiddenSize'],
                                   num_layers=args['dec_numlayer'])

        self.out_unit = nn.Linear(args['hiddenSize'], args['vocabularySize'])
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.element_len = args['hiddenSize']

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.CEloss = torch.nn.CrossEntropyLoss(reduction='none')
        self.NLLloss = torch.nn.NLLLoss()
        self.sigmoid = torch.nn.Sigmoid()

        self.init_state = (torch.rand(args['dec_numlayer'], 1, args['hiddenSize']).to(args['device']),
                           torch.rand(args['dec_numlayer'], 1, args['hiddenSize']).to(args['device']))
        self.M = Parameter(torch.rand(args['hiddenSize'], args['embeddingSize']))

    def getloss(self, decoderInputs, decoderTargetsEmbedding, decoderTargets, inputmask=None, eps=1e-20):
        '''
        :param decoderInputs:
        :param decoderTargetsEmbedding:  b s e
        :return:
        '''
        batch_size = decoderInputs.size()[0]
        dec_len = decoderInputs.size()[1]


        dec_input_embed = self.embedding(decoderInputs)
        # if mask is not None:
        #     dec_input_embed = dec_input_embed * mask.unsqueeze(2)

        init_state = (self.init_state[0].repeat([1, batch_size, 1]), self.init_state[1].repeat([1, batch_size, 1]))
        de_outputs, de_state = self.decoder_t(init_state, dec_input_embed, batch_size, dec_len)  # b s h
        temp1 = torch.einsum('bsh,he->bse', de_outputs, self.M)  # b s e
        temp2 = torch.einsum('bse,bse->bs', temp1, decoderTargetsEmbedding)
        probs = self.sigmoid(temp2)

        recon_loss = - torch.log(probs + eps)
        if inputmask is None:
            mask = torch.sign(decoderTargets.float())
            recon_loss = recon_loss * mask
        else:
            recon_loss = recon_loss * inputmask

        recon_loss_mean = torch.mean(recon_loss, dim=-1)
        return temp1, recon_loss_mean

    def buildModel(self, x, test=False, eps=1e-20):
        self.decoderInputs = x['dec_input']
        self.decoder_lengths = x['dec_len']
        self.decoderTargets = x['dec_target']
        decoderTargetsEmbeddings = self.embedding(self.decoderTargets)
        temp1, recon_loss_mean = self.getloss(self.decoderInputs, decoderTargetsEmbeddings, self.decoderTargets)
        fake_recon_loss = 0

        if not test:
            negatives = self.get_negative_samples(self.decoderTargets)  # b s 10
            negativeEmbeddings = self.embedding(negatives)  # b s 10 e
            fake_temp2 = torch.einsum('bse,bsne->bsn', temp1, negativeEmbeddings)
            fake_probs = self.sigmoid(fake_temp2)
            fake_recon_loss = - torch.log(fake_probs + eps)
            fake_recon_loss = torch.sum(fake_recon_loss, dim=2)  # b s
            fake_recon_loss = torch.sum(fake_recon_loss, dim=1)
        return recon_loss_mean, fake_recon_loss

    def get_negative_samples(self, indextensor, samplenum=10):
        '''
        :param indextensor:  b s
        :return: b s num
        '''
        t1 = time.time()
        batch = indextensor.size()[0]
        seqlen = indextensor.size()[1]
        weights = torch.tensor([1 for _ in range(args['vocabularySize'])], dtype=torch.float)
        samples = torch.multinomial(weights, batch * seqlen * samplenum, replacement=True)  # bs * 10
        res = samples.reshape((batch, seqlen, samplenum))
        # print(time.time() - t1)
        return res.to(args['device'])

    def decoder_t(self, initial_state, inputs, batch_size, dec_len):
        inputs = torch.transpose(inputs, 0, 1).contiguous()
        state = initial_state

        output, out_state = self.dec_unit(inputs, state)

        # output = self.out_unit(output.view(batch_size * dec_len, args['hiddenSize']))
        # output = output.view(dec_len, batch_size, args['vocabularySize'])
        output = torch.transpose(output, 0, 1)
        return output, out_state

    def forward(self, x, test=False):
        recon_loss_mean, fake_loss = self.buildModel(x, test)
        return recon_loss_mean, fake_loss

    def LMloss(self, sampled_hard, decoderInputs, eps=1e-6):
        batchsize = decoderInputs.size()[0]
        seqlen = decoderInputs.size()[1]
        decoderTargets = decoderInputs[:, 1:]
        decoderInputs = decoderInputs[:, :-1]
        # print("current decoder inputs in LMloss : ", decoderInputs)
        # print("current decoder Targets in LMloss : ", decoderTargets)
        decoderTargetsEmbeddings = self.embedding(decoderTargets)
        decoderTargetsEmbeddings = decoderTargetsEmbeddings * sampled_hard[:, 1:].unsqueeze(2)
        packed_input = torch.transpose(decoderInputs, 0, 1)
        mask = torch.transpose(sampled_hard[:, :-1], 0, 1)
        mask_out = torch.transpose(sampled_hard[:, 1:], 0, 1)
        target_embs = decoderTargetsEmbeddings.transpose(0, 1)
        packed_out = []
        hidden = (self.init_state[0].repeat([1, batchsize, 1]), self.init_state[1].repeat([1, batchsize, 1]))
        y_iter = torch.zeros(batchsize, args['embeddingSize']).to(args['device'])
        ys = []
        for ind in range(seqlen - 1)[::-1]:
            m = mask_out[ind, :].unsqueeze(-1)
            y = target_embs[ind, :, :]
            # print(target_embs.size(),y.size(), y_iter.size(), m.size())
            y_iter = m * y + (1 - m) * y_iter
            ys.append(y_iter)
        ys = torch.stack(ys[::-1])  # s b h

        packed_input_embs = self.embedding(packed_input)  # s b e
        for x, m in zip(packed_input_embs, mask):
            m = m.unsqueeze(-1)
            # print(x.size(), x.unsqueeze(0).size())
            x_post, (h1, c1) = self.dec_unit(x.unsqueeze(0), hidden)
            # print(mask[ind,:].size(), hidden1[0].size())
            h1 = m * h1 + (1 - m) * hidden[0]
            c1 = m * c1 + (1 - m) * hidden[1]
            hidden = (h1, c1)
            packed_out.append(x_post)
        packed_out = torch.cat(packed_out, dim=0).transpose(0, 1)
        # print(packed_out.size(), self.M.size())
        temp1 = torch.einsum('bsh,he->bse', packed_out, self.M)  # b s e
        temp2 = torch.einsum('bse,sbe->bs', temp1, ys)
        probs = self.sigmoid(temp2)

        recon_loss = - torch.log(probs + eps)
        # print(recon_loss.size(), mask.size())
        recon_loss = recon_loss * sampled_hard[:, :-1]
        # temp1, recon_loss_mean = self.getloss(decoderInputs, decoderTargetsEmbeddings, decoderTargets)
        recon_loss_mean = recon_loss.mean(1)
        return recon_loss_mean


def train(textData, model, model_path, print_every=10000, plot_every=10, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print(type(textData.word2index))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)

    iter = 1
    batches = textData.getBatches_forLM()
    n_iters = len(batches)
    print('niters ', n_iters)

    args['trainseq2seq'] = False

    epoch_loss_history = []
    ppl_history = []
    min_epoch_loss = 100000


    min_ppl = -1
    # accuracy = self.test('test', max_accu)
    for epoch in range(args['numEpochs']):
        print("currently starting epoch: ", epoch)
        losses = []

        for batch in batches:

            optimizer.zero_grad()
            x = {}
            x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs)).to(args['device'])
            x['dec_len'] = batch.decoder_lens
            x['dec_target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs)).to(args['device'])

            # print("x is: ",x)
            loss, fake_loss = model(x)  # batch seq_len outsize
            loss = torch.mean(loss - fake_loss)
            # print("loss is: ", loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])

            optimizer.step()

            print_loss_total += loss.data
            plot_loss_total += loss.data

            losses.append(loss.data)

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / (n_iters * args['numEpochs'])),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            iter += 1

        ppl = test(textData, model, 'test')
        # if ppl < min_ppl or min_ppl == -1:
        #     print('ppl = ', ppl, '<= min_ppl(', min_ppl, '), saving model...')
        #     torch.save(model, model_path)
        #     min_ppl = ppl
        epoch_loss = sum(losses) / len(losses)
        if epoch_loss < min_epoch_loss:
            print(f"current epoch loss: {epoch_loss}")
            print('ppl = ', ppl, '<= min_ppl(', min_ppl, '), saving model...')
            torch.save(model, model_path)
            min_epoch_loss = epoch_loss



        print('Epoch ', epoch, 'loss = ', sum(losses) / len(losses), 'Valid ppl = ', ppl,
              'min ppl=', min_ppl)

        epoch_loss_history.append((sum(losses) / len(losses)).item())
        ppl_history.append(ppl)

    training_stats = {}
    training_stats['epoch'] = list(range(len(epoch_loss_history)))
    training_stats['epoch_loss'] = epoch_loss_history
    training_stats['ppl'] = ppl_history

    pd.DataFrame(training_stats).to_csv(
        args['rootDir'] + "/LM_training_stats_" + args['date'] + ".csv",
        index=False)
    # self.test()
    # showPlot(plot_losses)


def test(textData, model, datasetname, eps=1e-20):
    ave_loss = 0
    num = 0
    with torch.no_grad():
        for batch in textData.getBatches_forLM(datasetname):
            x = {}
            x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs)).to(args['device'])
            x['dec_len'] = batch.decoder_lens
            x['dec_target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs)).to(args['device'])
            recon_loss_mean, _ = model(x, test=True)  # batch seq_len outsize
            ave_loss = (ave_loss * num + sum(recon_loss_mean)) / (num + len(recon_loss_mean))
            num += len(recon_loss_mean)

    return torch.exp(ave_loss)


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g')
    parser.add_argument('--modelarch', '-m')
    cmdargs = parser.parse_args()
    usegpu = True
    if cmdargs.gpu is None:
        usegpu = False
    else:
        usegpu = True
        args['device'] = 'cuda:' + str(cmdargs.gpu)


def kenlm_test(textData):
    import kenlm
    model = kenlm.Model('/Users/shalei/科研/2020/kenlm/beer.lm')
    ppls = []
    for batch in textData.getBatches_forLM('test'):
        sentences = [' '.join([textData.index2word[wid] for wid in r if wid > 3]) for r in batch.decoderSeqs]
        # print(sentences[0])
        ppl = [np.log(model.perplexity(s)) for s in sentences]
        ppls.extend(ppl)

    return sum(ppls) / len(ppls)


if __name__ == '__main__':
    textData = TextDataMimic("mimic", "../clinicalBERT/data/",  "discharge", trainLM=True, test_phase=False, big_emb = False, new_emb=False)
    # nll = kenlm_test(textData)
    # print('LM=', nll, np.exp(nll))

    parseargs()
    args['batchSize'] = 128
    # args['maxLength'] = 1000
    # args['maxLengthEnco'] = args['maxLength']
    # args['maxLengthDeco'] = args['maxLength'] + 1
    args['vocabularySize'] = textData.getVocabularySize()
    args['chargenum'] = 0
    args['embeddingSize'] = textData.index2vector.shape[1]

    model = LanguageModel(textData.word2index, textData.index2word, textData.index2vector).to(args['device'])
    train(textData, model, model_path = args['rootDir']+'/LMmimic.pkl')
    print("using device: ", args['device'])

