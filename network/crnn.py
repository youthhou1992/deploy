import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torch.autograd import Variable

class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    #                   32    1   37     256
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        nm = [64, 64, 128, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=True):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        convRelu(1)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(2)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(6, True)
        convRelu(7)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(8, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        #print('---forward propagation---')
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = F.log_softmax(self.rnn(conv), dim=2)
        return output
#h*batch_size*nclass
#        return output
class resizeNormalize(object):

    def __init__(self, interpolation=Image.BILINEAR):
        self.interpolation = interpolation
        self.toTensor = T.ToTensor()
    def __call__(self, img, size):
        img = img.resize(size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index
        # self.alphabet = "".join(alphabet.split("\n")) + '\n'+ "-"


        # self.dict = {}
        # for i, char in enumerate(alphabet):
        #     # NOTE: 0 is reserved for 'blank' required by wrap_ctc
        #     self.dict[char] = i + 1

    def decode(self, t, length=0, is_pred_60=False, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            if is_pred_60:
                # t = torch.cat([t, torch.tensor([0])])
                # index = t - t.roll(shifts=1)
                index = torch.diff(t)
                pred_idx = t[index.nonzero()].view(-1)
            else:
                pred_idx = t.view(-1)
            pred_idx = pred_idx[(pred_idx % 2).nonzero().view(-1)] 
            char_list = [self.alphabet[idx-1] for idx in pred_idx] 
            return ''.join(char_list)

        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            # texts = []
            # append = texts.append
            batch_size = length.shape[0]
            # b = torch.cat([t.reshape((batch_size, -1)), torch.tensor([[0]] * batch_size, device="cuda")], dim=1)
            # index = b - b.roll(shifts=1)
            b = t.reshape((batch_size, -1))
            index = torch.diff(b)
            leng = torch.IntTensor([1])
            idx_nonzero = index.nonzero()
            # for i in range(batch_size):
            #     pred_idx = b[i][idx_nonzero[idx_nonzero[:,0]==i][:,1]]
            #     append(self.decode(pred_idx, length))
            texts = [self.decode(b[i][idx_nonzero[idx_nonzero[:,0]==i][:,1]], leng) for i in range(batch_size)]
            # index = 0
            # for i in range(length.numel()):
            #     l = length[i]
            #     append(
            #         self.decode(
            #             t[index:index + l], torch.IntTensor([l]), raw=raw))
            #     index += l
            return texts


def crnn_rec(rec_im, model, converter, is_pred_60):
    """
        Args:
            rec_ims: list of images for crnn recognition
            model: crnn model
            alphabet: import alphabets; alphabet = alphabets.alphabet
    """

    model.eval()
    if is_pred_60[0]:
        rec_im = Image.fromarray(rec_im)
        size = (int(1. * rec_im.size[0] * 32 / rec_im.size[1]), 32)
        # size = (480, 32)
        rec_im = rec_im.convert('L')
        transformer = is_pred_60[1]
        rec_im = transformer(rec_im, size)
        if torch.cuda.is_available():
            rec_im = rec_im.cuda()
        rec_im = rec_im.view(1, *rec_im.size())
        rec_im = Variable(rec_im)

    # gpu_tracker.track()
    if torch.cuda.is_available():
        rec_im = rec_im.cuda()
        
    with torch.no_grad():
        preds = model(rec_im)
    # gpu_tracker.track()
    # ooo = preds.cpu().detach().numpy()
    # ooo = np.squeeze(ooo)
    # probs = np.max(ooo, axis=1)
    # probs = np.sum(probs)
    # conf = np.exp(probs)
    # conf = [conf]
    # print(preds.shape)
    probs = torch.max(preds, dim=2).values
    conf = torch.exp(torch.sum(probs, dim=0))

    _, predss = preds.max(2)
    preds = predss.transpose(1, 0).contiguous().view(-1)
    # if is_pred_60:
    #     preds_size = Variable(torch.IntTensor([predss.size(0)]))
    # else:
    preds_size = Variable(torch.IntTensor([predss.size(0)] * predss.size(1)))
    pred = converter.decode(preds.data, preds_size.data, is_pred_60[0], raw=False)
    return pred, conf