import sys

class NoEmptyPercentage:
    def __init__(self, input, output, fieldName):
        self.input = input
        self.output = output
        self.fieldName = fieldName

    def run(self):
        inputFile = open(self.input, 'r', encoding='utf-8')
        outputFile = open(self.output, 'w', encoding='utf-8')

        totalLine = 0
        noEmptyLine = 0

        schema = {}
        isFirst = True
        for line in inputFile:
            fields = line[:-1].split('\t')
            if isFirst:
                for i in range(len(fields)):
                    schema[fields[i]] = i

                isFirst = False
                continue

            totalLine += 1
            if (self.fieldName in schema and fields[schema[self.fieldName]]):
                noEmptyLine += 1

        outputFile.write(str(noEmptyLine / totalLine) + '\n')

          
if __name__ == '__main__':
    process = NoEmptyPercentage(sys.argv[1], sys.argv[2], sys.argv[3])
    #process = NoEmptyPercentage("D:\Test\OSearchTest\Precision.txt", "D:\Test\OSearchTest\output.txt", "m:pbaEntity")
    process.run()
