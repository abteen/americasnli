import sys, os, re

if __name__ == '__main__':

    main_dir = sys.argv[1]
    split = sys.argv[2].capitalize()
    langs = ['aym', 'bzd', 'cni', 'gn', 'hch', 'nah', 'oto', 'quy', 'shp', 'tar']
    files = os.listdir(main_dir)

    scores = []
    for lang in langs:
        if lang+'.log' not in files:
            scores.append((lang, 0.0))
        else:
            with open(os.path.join(main_dir, lang) + '.log', 'r') as f:
                score = 0.0
                for line in f:
                    if '{} accuracy'.format(split) in line:
                        score = float(re.findall(r'[0-9][0-9]\.[0-9]+$', line)[0])
                scores.append((lang.replace('.log', ''), score))

    langs, scores = zip(*scores)

    print(langs)

    output = ''
    for score in scores:
        output += ',{:.2f}'.format(score)
    print(output)