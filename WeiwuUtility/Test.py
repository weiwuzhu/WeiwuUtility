import sys
import re

class KeywordExtractor:
    def __init__(self, input, output, newFieldName, hasHeader=True):
        self.input = input
        self.output = output
        self.newFieldName = newFieldName
        self.hasHeader = hasHeader

    def run(self):
        inputFile = open(self.input, 'r', encoding='utf-8')
        outputFile = open(self.output, 'w', encoding='utf-8')

        isFirst = True
        for line in inputFile:
            fields = line[:-1].split('\t')
            if isFirst and self.hasHeader:
                outputFile.write(line[:-1] + '\t' + self.newFieldName + '\n')
                isFirst = False
                continue

            query = fields[0].lower()
            queryTerms = set()
            for token in filter(None, query.split(' ')):
                queryTerms.add(token)
            feedsQuery = fields[-1].lower()
            keywords = feedsQuery
            index = keywords.rfind('[nqlf_base16$catvec')
            if index != -1:
                keywords = keywords[:index].strip()
            index = keywords.rfind(']')
            if index != -1:
                keywords = keywords[index+1:].strip()

            start = 0
            i = 0
            while i < len(keywords):
                if keywords[i] == ' ':
                    token = keywords[start:i]
                    if token in queryTerms or token.startswith('word:') or token.startswith('rankonly:') or token.startswith('addquery:'):
                        break;
                    start = i + 1
                i += 1

            keywords = keywords[start:]
            
            pattern = '\[qlf\$1797\:([0-9]+)\]'
            BTCs = []
            for match in re.findall(pattern, feedsQuery):
                BTCs.append(match)
                
            BTC = '' if len(BTCs) == 0 else BTCs[0]
            
            outputFile.write(line[:-1] + '\t' + keywords + '\t' + BTC +'\n')

          
if __name__ == '__main__':
    sr = KeywordExtractor(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    #sr = KeywordExtractor("D:\Test\FeedQuery.txt", "D:\Test\output1.txt", "m:KeyWords\tm:BTC", 1)
    sr.run()
