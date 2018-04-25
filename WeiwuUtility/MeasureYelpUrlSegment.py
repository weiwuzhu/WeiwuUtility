import sys
import re

class YelpUrlSegmentMeasurement:
    def __init__(self, input, output, hasHeader=True):
        self.input = input
        self.output = output
        self.hasHeader = hasHeader

    def run(self):
        inputFile = open(self.input, 'r', encoding='utf-8')
        outputFile = open(self.output, 'w', encoding='utf-8')
        
        result = []
        
        schema = {}
        isFirst = True
        for line in inputFile:
            fields = line[:-1].split('\t')
            if isFirst and self.hasHeader:
                for i in range(len(fields)):
                    schema[fields[i]] = i
                outputFile.write(line[:-1] + '\tIsMatch\n')
                isFirst = False
                continue

            truth = fields[schema['Truth']].lower()
            predictSegmentName = fields[schema['SegmentName']].lower()
            isMatch = False
            if predictSegmentName:
                for item in predictSegmentName.split(';'):
                    if item in truth:
                        isMatch = True
            
            outputFile.write(line[:-1] + '\t' + str(isMatch) +'\n')
            if fields[schema['Url']]:
                result.append([predictSegmentName, truth, isMatch])
        
        segments = set()
        for [predictSegmentName, truth, isMatch] in result:
            if predictSegmentName:
                for item in predictSegmentName.split(';'):
                    segments.add(item)
                    
        stats = {}
        for [predictSegmentName, truth, isMatch] in result:
            if predictSegmentName:
                for item in predictSegmentName.split(';'):
                    if item not in stats:
                        stats[item] = [0, 0, 0]
                    stats[item][1] += 1
                    
            if truth:
                for item in truth.split(";||#"):
                    if item in segments:
                        if item not in stats:
                            stats[item] = [0, 0, 0]
                        stats[item][0] += 1
                        if isMatch:
                            stats[item][2] += 1
                
        outputFile2 = open("D:\Test\output3.txt", 'w', encoding='utf-8')
        for segment in stats:
            outputFile2.write(segment + '\t' + str(stats[segment][0]) + '\t' + str(stats[segment][1]) + '\t' + str(stats[segment][2]) + '\n')

          
if __name__ == '__main__':
    #worker = YelpUrlSegmentMeasurement(sys.argv[1], sys.argv[2], sys.argv[3])
    worker = YelpUrlSegmentMeasurement("D:\Test\YelpUrlSegmentTest.txt", "D:\Test\output2.txt", 1)
    worker.run()
