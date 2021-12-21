f = open('in_domain_dev.tsv', 'r').readlines()
g1 = open('val.src', 'w')
g2 = open('val.tgt', 'w')

for line in f:
    line = line.split('\t')
    _, label, _, text = line

    g1.write('{}'.format(text))
    g2.write('{}\n'.format(label))
