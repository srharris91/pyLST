import numpy as np
import re
import json
def find_nearest_arg(array:np.ndarray,value:np.complex) -> int:
    ''' Returns the nearest index in array nearest value
    Inputs:
        array:np.ndarray to search
        value:np.complex value to find nearest index
    Returns:
        index:int index in array that is closest to the desired value
    '''

    return (np.nanargmin(np.abs(array-value)))

def ifkey(dic:dict,key):
    ''' Return the value of dic[key] if it exists, else False
    '''

    if key in dic:
        return dic[key]
    else:
        return False

def ifflag(params:dict,flag):
    ''' Return the value of params['flags'][flag] if it exists, else False
    '''

    if ifkey(params,'flags'):
        return ifkey(params['flags'],flag)
    else:
        return False

def removeCCppComment( text:str ) -> str :
    ''' Remove c++ comments like " // comment " and "/* comment */", uses re module
        
        Inputs:
            text:str text to remove c++ comments from
        Returns:
            str containing text with removed comments
    '''

    def blotOutNonNewlines( strIn ) :  # Return a string containing only the newline chars contained in strIn
        return "" + ("\n" * strIn.count('\n'))

    def replacer( match ) :
        s = match.group(0)
        if s.startswith('/'):  # Matched string is //...EOL or /*...*/  ==> Blot out all non-newline chars
            return blotOutNonNewlines(s)
        else:                  # Matched string is '...' or "..."  ==> Keep unchanged
            return s

    pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )

    return re.sub(pattern, replacer, text)

def read_jsoncpp(filename):
    ''' read json file (may contain c++ type comments)
    '''

    with open(filename,'r') as f: 
        string_data = ''.join([row for row in f.readlines()])
        string_data = removeCCppComment(string_data)
        data = json.loads(string_data)
    return data

