from sys import argv
from copy import *
from math import log

class Node:

    def __init__(self, label):
        # for non-leafs it is the name of the attribute
        # for leafs it is the class
        self.label = label
        
        # dictionary of (attribute value, node)
        self.children = {}
    
    def display(self, string):
        print(string + self.label)
        string += "\t"
        if self.children:
            for key, value in self.children.items():
                print(string + key)
                value.display(string + "\t")


def getDataSet(dataSetName):
    # open a file and read the data: classes, attributes and examples
    f = open(dataSetName, 'r')

    # get attribute names
    line = f.readline().split()
    attributeNames = line[1:]
    attributes = {}
    for a in line[1:]:
        attributes[a] = set()
    
    classes = set()
    
    # read examples        
    examples = []
    line = f.readline().split()
    while line:
        classValues = line[0]
        classes.add(classValues)
        
        example = {}
        example["CLASS"] = classValues
        for i in range(1, len(line)):
            # attribute
            attributeName = attributeNames[i - 1]
            attributeValue = line[i]
            attributes[attributeName].add(attributeValue)

            # example
            example[attributeName] = attributeValue
            
        examples.append(example)
        line = f.readline().split()
    
    f.close()
    
    return classes, attributes, examples


def mostFrequentClass(examples, classes):
    return None


def entropy(examples, classes):
    counter = {}

    for c in classes:
        counter[c] = 0

    for c in classes:
        for elem in examples:
            if(elem["CLASS"] == c):
                counter[c] += 1
    total = 0
    for k,v in counter.items():
        total += v


    for c in classes:
        if counter[c] == 0.0:
            return 0

    ent = 0.0
    for c,v in counter.items():
        ent += -(float(v)/total)*log((float(v)/total), 2)

    return ent


def gain(examples, classes, atribut):
    E = entropy(examples, classes)

    valAttr = []

    for l in examples:
        valAttr.append(l[atribut])

    result = {}
    for word in valAttr:
        if word in result:
            result[word] += 1
        else:
            result[word] = 1

    for k in result.keys():
        tempSet = []
        for l in examples:
            if l[atribut] == k:
                tempSet.append(l)
        E = E - (len(tempSet) * 1.0 / len(examples) * 1.0) * entropy(tempSet, classes)

    return E



def ID3(examples, classes, attributes):
    counter = {}

    for c in classes:
        counter[c] = 0

    for c in classes:
        for elem in examples:
            if (elem["CLASS"] == c):
                counter[c] += 1

    #toate frunzele aparin unei clase
    diff_zero = 0
    max_key = None
    max_value = None
    for k,v in counter.items():
        if v > 0:
            diff_zero += 1
            max_key = k
            max_value = v

    if diff_zero == 1:
        return Node(max_key)

    #nu exista atribute
    if attributes == {}:
        key, value = max(counter.items(), key=lambda x: x[1])
        return Node(key)

    #gasim atributul cu cel mai mare castig informational
    gains = {}
    for k,v in attributes.items():
        gains[k] = gain(examples, classes, k)

    max_key, max_value = max(gains.items(), key=lambda x: x[1])

    nod = Node(max_key)
    #create new examples and attributes
    new_attributes = {}
    for k,v in attributes.items():
        if k != max_key:
            new_attributes[k] = v

    for a in attributes[max_key]:
        new_examples = []
        for exem in examples:
            if exem[max_key] == a:
                new_examples.append(exem)
        nod.children[a] = ID3(deepcopy(new_examples), classes, deepcopy(new_attributes))

    return nod


def evaluate(tree, example, classes):
    while True:
        atribut = tree.label
        atribut_value = example[atribut]
        tree = tree.children[atribut_value]
        if tree.label in classes:
            return tree.label

if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: " + argv[0] + " dataset_name")
        exit()
               
    # classes is a set of all posible classes
    # attributes is a dictionary of sets: (attribute name, set of attribute values)        
    # examples is a list of examples
    # an example is dictionary: (class, attribute1, attribute2, ...)
    classes, attributes, examples = getDataSet(argv[1])
    print("Classes")
    print(classes)
    print("\n")

    print("Attributes")
    print(attributes)
    print("\n")

    print("Examples")
    print(examples)
    print("\n")
    

    print("Decision tree")
    tree = ID3(examples, classes, attributes)
    tree.display("")

    for exem in examples:
        print(exem["CLASS"], evaluate(tree, exem, classes))
    # evaluate the decision tree on the examples
                 
                  
              
