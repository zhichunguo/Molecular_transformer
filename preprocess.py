''' Handling the data io '''
import argparse
import torch
import transformer.Constants as Constants

mole_dict = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: " Ne",
                11: "Na", 12:"Mg", 13: "Al", 14:"Si", 15:"P", 16: "S", 17: "Cl", 18:"Ar", 19:"K", 20:"Ca", 22:"Ti", 24:"Cr", 26:"Fe", 28:"Ni",
                29:"Cu", 31:"Ga", 32:"Ge", 34:"Se", 35:"Br", 40:"Zr", 44:"Ru", 45:"Rh", 46:"Pd", 47:"Ag", 50:"Sn", 51:"Sb", 52:"Te", 53: "I", 65:"Tb", 75:"Re", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg",
                81:"Tl", 82:"Pb", 83:"Bi"}

pair_list = ["br", "cl", "si", "na", "ca", "ge", "cu", "au", "sn", "tb", "pt", "re", "ru", "bi", "li", "fe", "sb", "hg","pb", "se", "ag","cr","pd","ga","mg","ni","ir","rh","te","ti","al","zr","tl", "nh"]

def read_instances_from_file(inst_file, max_sent_len, keep_case):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    labels = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            mol, label = sent.strip('\n').split('\t')

            words = []
            i = 0
            while i < len(mol):
                if mol[i: i+2] in pair_list:
                    words.append(mol[i: i+2])
                    i += 2
                else:
                    words.append(mol[i])
                    i += 1

            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = words[:max_sent_len]

            if word_inst:
                word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:
                word_insts += [None]

            labels.append([float(label)])

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts, labels

def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurrence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK,
        Constants.VAE_WORD: Constants.VAE}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx, tgt):
    ''' Mapping words to idx sequence. '''
    if tgt:
        return [word2idx.get(s[1], Constants.UNK) for s in word_insts]
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    # parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-valid_src', required=True)
    # parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-save_data', required=False)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=100)
    parser.add_argument('-min_word_count', type=int, default=3)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # include the <s> and </s>

    # Training set
    train_word_insts, train_labels = read_instances_from_file(opt.train_src, opt.max_word_seq_len, opt.keep_case)

    #- Remove empty instances
    train_word_insts, train_labels = list(zip(*[
        (s, t) for s, t in zip(train_word_insts, train_labels) if s and t]))

    # Validation set
    valid_word_insts, valid_labels = read_instances_from_file(opt.valid_src, opt.max_word_seq_len, opt.keep_case)

    #- Remove empty instances
    valid_word_insts, valid_labels = list(zip(*[
        (s, t) for s, t in zip(valid_word_insts, valid_labels) if s and t]))

    src_word2idx = build_vocab_idx(train_word_insts, opt.min_word_count)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_word_insts, src_word2idx, tgt=False)
    valid_src_insts = convert_instance_to_idx_seq(valid_word_insts, src_word2idx, tgt=False)

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': src_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_labels},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_labels}
        }

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
