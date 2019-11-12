from modules.bidaf_bert_utils import Linear, LSTM
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class BiDAFBlock(nn.Module):
    """Thanks to Taeuk Kim's repo https://github.com/galsang/BiDAF-pytorch
    We make the following modification:
        delete charater & word embedding layers
        delete contextual layer
        assume bert will gives us the embedding and masking to do the prediction
    """
    def __init__(self, config):
        super(BiDAFBlock, self).__init__()
        self.config = config
        # 3. Contextual Embedding Layer is BERT, which we assume will give BiDAF output
        self.SEP = 102
        # 4. Attention Flow Layer
        self.att_weight_c = Linear(config.hidden_size, 1)
        self.att_weight_q = Linear(config.hidden_size, 1)
        self.att_weight_cq = Linear(config.hidden_size, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=config.hidden_size * 4,
                                   hidden_size=config.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=config.dropout)

        self.modeling_LSTM2 = LSTM(input_size=config.hidden_size * 2,
                                   hidden_size=config.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=config.dropout)

        # 6. Output Layer
        self.p1_weight_g = Linear(config.hidden_size * 4, 1, dropout=config.dropout)
        self.p1_weight_m = Linear(config.hidden_size * 2, 1, dropout=config.dropout)
        self.p2_weight_g = Linear(config.hidden_size * 4, 1, dropout=config.dropout)
        self.p2_weight_m = Linear(config.hidden_size * 2, 1, dropout=config.dropout)

        self.output_LSTM = LSTM(input_size=config.hidden_size * 2,
                                hidden_size=config.hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=config.dropout)

        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, input_tensors, input_ids):
        """

        :param input_tensors: bert output [batch_size, sentence_len, embedding_size]
        :param token_mask: 1 for token, 0 for padding
        :param input_ids: used for find [SEP] and seperate question and answer
        :return:
        """
        # TODO: More memory-efficient architecture
        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size)
            :param q: (batch, q_len, hidden_size)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            cq = []
            for i in range(q_len):
                #(batch, 1, hidden_size)
                qi = q.select(1, i).unsqueeze(1)
                #(batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size) -> (batch, c_len, hidden_size)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len,output_layer(g, m, c_lens) hidden_size) -> (batch, hidden_size)
            q2c_att = torch.bmm(b, c).squeeze()
            # (batch, c_len, hidden_size) (tiled)
            q2c_att = q2c_att.unsqueeze(1)  # .expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 4)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch, c_len, hidden_size * 4)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l))[0]
            # (batch, c_len)
            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()

            return p1, p2

        def qc_from_bert(input_tensors, input_ids, sep_idx: int):
            # remove cls token
            # [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            input_tensors = input_tensors[:, 1:, :]
            sep_token_mask = 1 - 1 * (input_ids != sep_idx)  # 1 for sep, 0 for nonsep
            q_tensors = []
            c_tensors = []
            q_lens = []
            c_lens = []
            max_q_len = 0
            max_c_len = 0
            for i in range(len(sep_token_mask)):
                t = sep_token_mask[i]
                sep_loc = t.nonzero()
                assert len(sep_loc) == 2  # there should be only 2 [SEP] tokens in the each data points
                q_tensor = input_tensors[i, :sep_loc[0]]
                c_tensor = input_tensors[i, sep_loc[0] + 1:sep_loc[1]]

                # record the length of q
                q_lens.append(len(q_tensor))
                c_lens.append(len(c_tensor))

                max_q_len = max(len(q_tensor), max_q_len)
                max_c_len = max(len(q_tensor), max_c_len)
                q_tensors.append(q_tensor)
                c_tensors.append(c_tensor)
            c_lens = torch.tensor(c_lens)
            q_lens = torch.tensor(q_lens)
            c_tensors = pad_sequence(c_tensors, batch_first=True, padding_value=0)
            q_tensors = pad_sequence(q_tensors, batch_first=True, padding_value=0)
            return q_tensors, c_tensors, q_lens, c_lens

        # 1. Contextual Embedding from bert
        q, c, q_lens, c_lens = qc_from_bert(input_tensor, input_ids, self.SEP)
        # 2. Attention Flow Layer
        g = att_flow_layer(c, q)
        # 3. Modeling Layer
        m = self.modeling_LSTM2((self.modeling_LSTM1((g, c_lens))[0], c_lens))[0]
        # 4. Output Layer
        p1, p2 = output_layer(g, m, c_lens)

        # (batch, c_len), (batch, c_len)
        return p1, p2


if __name__ == "__main__":
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-dim', default=8, type=int)
    parser.add_argument('--char-channel-width', default=5, type=int)
    parser.add_argument('--char-channel-size', default=100, type=int)
    parser.add_argument('--context-threshold', default=400, type=int)
    parser.add_argument('--dev-batch-size', default=100, type=int)
    parser.add_argument('--dev-file', default='dev-v1.1.json')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=12, type=int)
    parser.add_argument('--exp-decay-rate', default=0.999, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=768, type=int)
    parser.add_argument('--learning-rate', default=0.5, type=float)
    parser.add_argument('--print-freq', default=250, type=int)
    parser.add_argument('--train-batch-size', default=60, type=int)
    parser.add_argument('--train-file', default='train-v1.1.json')
    parser.add_argument('--word-dim', default=100, type=int)
    args = parser.parse_args()
    model = BiDAFBlock(args)
    import pickle
    input_tensor = pickle.load(open("seq_output.pt", "rb"))
    input_mask = pickle.load(open("attention_mask.pt", "rb"))
    input_ids = pickle.load(open("input_ids.pt", "rb"))
    model(input_tensor, input_ids)
    print()
