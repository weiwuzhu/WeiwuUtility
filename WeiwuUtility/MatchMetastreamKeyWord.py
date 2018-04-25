import sys
import re

class MatchMetastreamKeyWord:
    def __init__(self, input, output, metastreams, hasHeader=True):
        self.input = input
        self.output = output
        self.metastreams = metastreams
        self.hasHeader = hasHeader
        self.metastreamList = [x.strip() for x in metastreams.split(',')]

    def parseKeywords(self, keywords):
        keywordList = []
        keywords += ' '
        start = 0
        i = 0
        while i < len(keywords):
            if keywords[i] == ' ':
                token = keywords[start:i]
                if not token or token.isspace():
                    start = i + 1
                elif token.startswith('word:'):
                    if token[-1] == ')':
                        item = []
                        wordstr = token[6:-1]
                        keywordList2 = self.parseKeywords(wordstr)
                        for m in keywordList2:
                            item.append(m[0])
                        keywordList.append(item)
                        start = i + 1
                elif token.startswith('rankonly:'):
                    if token.startswith('rankonly:(') or token.startswith('rankonly:word:('):
                        if token[-1] == ')':
                            start = i + 1
                    else:
                        start = i + 1
                elif token.startswith('"'):
                    if token[-1] == '"':
                        item = []
                        item.append(token[1:-1])
                        keywordList.append(item)
                        start = i + 1
                else:
                    item = []
                    item.append(token)
                    keywordList.append(item)
                    start = i + 1
            i += 1
        return keywordList
        
    def run(self):
        inputFile = open(self.input, 'r', encoding='utf-8')
        outputFile = open(self.output, 'w', encoding='utf-8')

        schema = {}
        isFirst = True
        for line in inputFile:
            fields = line[:-1].split('\t')
            if isFirst and self.hasHeader:
                for i in range(len(fields)):
                    schema[fields[i]] = i
                outputFile.write(line)
                isFirst = False
                continue

            keywords = fields[schema['m:Keywords']]
            index = keywords.find('addquery:')
            if index != -1:
                keywords = keywords[:index]
            keywordList = self.parseKeywords(keywords)
            
            for metastream in self.metastreamList:
                column = fields[schema[metastream]].lower()
                items = re.split('\W+', column)
                wordset = set()
                for token in filter(None, items):
                    wordset.add(token)
                isMatch = True
                unMatched = []
                for words in keywordList:
                    if len(words) > 1:
                        isSingleMatch = False
                        for word in words:
                            if len(list(filter(None, word.split(' ')))) > 1:
                                if word in column:
                                    isSingleMatch = True
                            else:
                                if word in wordset:
                                    isSingleMatch = True
                        if not isSingleMatch:
                            isMatch = False
                    else:
                        word = words[0]
                        if len(list(filter(None, word.split(' ')))) > 1:
                            if word not in column:
                                isMatch = False
                        else:
                            if word not in wordset:
                                isMatch = False
                    
                    if not isMatch:
                        unMatched.append(words)
                fields[schema[metastream]] = str(isMatch) if isMatch else str(unMatched)
            
            
            outputFile.write('\t'.join(fields) + '\n')

          
if __name__ == '__main__':
    sr = MatchMetastreamKeyWord(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    #sr = MatchMetastreamKeyWord("D:\\Test\\test.txt", "D:\\Test\\output.txt", "OdpTitle,QueryClick,OdpDescription,OdpCombined,FeedsMulti5,FeedsMulti8,FeedsMulti9", 1)
    sr.run()



