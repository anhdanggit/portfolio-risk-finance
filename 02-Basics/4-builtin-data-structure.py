'''
#FILE: BUILT-IN DATA STRUCTURE AND FUNCTION TRICKS
Project: Basic concepts in Python
-------------------
By: Anh Dang
Date: 2019-07-17
Description:
Some illustrations for basic concepts in Python
'''


## Tuple ---------
# Capture arbitrarily long list
values = 1,2,3,4,5
a, b, *rest = values
# Discard the rest
a, b, *_ = values

## Zip -------------
# "pairs" up the elements of list
seq1 = ['foo','bar','baz']
seq2 = ['one','two','three']

for i in list(zip(seq1, seq2)):
    a,b = i
    print('{}-{}'.format(b,a))

# Unzip a sequence
pitchers = [('Nolan','Ryan'),
            ('Roger','Clemens'),
            ('Schilling','Curt')]

first_names, last_name = zip(*pitchers)

# Create Dicts from Sequences
mapping = {}
key_list = ['Apple','Orange','Banana']
value_list = [1, 4, 8]
for key, value in zip(key_list, value_list):
    mapping[key] = value

# Categorizing to dicts
words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letters = {}

for word in words:
    letter = word[0]
    by_letters[letter] = by_letters.get(letter, 0) + 1
    value = by_letters[letter]
    print('Letter: {} - Count: {}'.format(letter, value))

by_letters = {}
for word in words:
    letter = word[0]
    by_letters.setdefault(letter, []).append(word)
print(by_letters)

# Empty Dict that ready-to-set default values
from collections import defaultdict
by_letters = defaultdict(list)
for word in words:
    by_letters[word[0]].append(word)
    print(by_letters)


## Functions -----------
## Multiple return
def f():
    a = 5
    b = 6
    c = 7
    return a, b, c

a, b, c = f()
print(a)
print(c)

## Alternative Option
def f():
    a = 5
    b = 6
    c = 7
    return {'a' : a, 'b' : b, 'c' : c}

print(f())


## Treated functions as object
import re

def remove_punctuation(value):
    return re.sub('[!#?]', '', value)

## Put functions by name as a list
clean_ops = [str.strip, remove_punctuation, str.title]
states = ['   Alabama ', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda', 
            'south   carolina##', 'West virginia?']

## Combined / Generic function
def clean_string(strings, ops):
    out = []
    for s in strings:
        for function in ops:
            s = function(s)
        out.append(s)
    return out 

clean_string(states, clean_ops)

## Use function name as a argument
list(map(remove_punctuation, states))

## Annoynymous lambda function
list(map(lambda s: remove_punctuation(str.strip(s)), states))

strings = ['foo', 'card', 'bar', 'aaaa', 'abab']
strings.sort(key=lambda x: len(set(list(x))))
print(strings)


## Currying: Partial Argument Application
'''Currying is computer science jargon (named after the mathematician Haskell Curry) 
that means deriving new functions from existing ones by partial argument application'''
def add_number(a, b):
    return a + b

add_5_to = lambda b: add_number(5, b)

add_number(2, 5)
add_5_to(2)

from functools import partial
add_5_to2 = partial(add_number, a=5)
add_5_to2(b=2)

## Try Catch 
def attempt_float(x):
    try:
        return float(x)
    except: ## maybe only ignore some (ValueError, TypeError)
        return 'Error'
    finally: ## try block success or not run this
        print('Done anyway')

attempt_float(6)
attempt_float('hello') ## ValueError
attempt_float((1,2)) ## TypeError


## Comprehensive -----
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
[x.upper() for x in strings if len(x) > 2] ## list comprehensive
{x.upper() for x in strings if len(x) > 2} ## set comprehensive
{key: val.upper() for key,val in enumerate(strings)} ## dict comprehensive

## Nested Comprehensive ----
all_data = [['John', 'Emily', 'Michael', 'Mary', 'Steven'],
            ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar', 'Eerin']]
## full
name_choose = []
for names in all_data:
    choose = [name for name in names if name.lower().count('e') >=2]
    name_choose.extend(choose)
name_choose
## comprehense
[name for names in all_data for name in names if name.lower().count('e') >=2]

## Use to flatten
some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
[i for tup in some_tuples for i in tup]
[[i for i in tup] for tup in some_tuples]


## Generator -----
'''Generator is to create an iterable objects'''
[x**2 for x in range(10)]
sum(x**2 for x in range(10))
dict((k, k**2) for k in range(5))

## itertools
import itertools

first_letter = lambda x: x[0]
names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']
names.sort() ## only working on consecutive elements
for letter, name in itertools.groupby(names, first_letter):
    print(letter, list(name))


