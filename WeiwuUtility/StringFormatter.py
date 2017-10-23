import re
import sys

class StringFormatter:
    def __init__(self, input, output, template, newFieldName, hasHeader=True):
        self.input = input
        self.output = output
        self.template = template
        self.newFieldName = newFieldName
        self.hasHeader = hasHeader

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

                newHeader = line
                if self.newFieldName not in schema:
                    newHeader = newHeader[:-1] + '\t' + self.newFieldName + '\n'
                outputFile.write(newHeader)

                isFirst = False
                continue

            pattern = '\{[0-9]+\}'
            s = self.template
            result = []
            for match in re.findall(pattern, self.template):
                s = s.replace(match, fields[int(match[1:-1])])

            newLine = line[:-1]
            if self.newFieldName in schema:
                fields[schema[self.newFieldName]] = s
                newLine = '\t'.join(fields)
            else:
                newLine += '\t' + s

            outputFile.write(newLine + '\n')

          
if __name__ == '__main__':
    #process = StringFormatter(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    process = StringFormatter("D:\Test\OSearchTest\ScrapingInput.txt", "D:\Test\OSearchTest\output.txt", '{0}{AugURLBegin#p1=[AnswerReducer mode="KeepOnlySelected" answers="Dolphin,PbaInstrumentation,OpalInstrumentation,LocationExtractorV2,Local"][APlusAnswer EntitiesDebug="true"][OSearch xapdebug="1"]&location=lat:{1};long:{2};disp:","#AugURLEnd}', "m:QueryText", 1)
    process.run()
