import string

def textstrip(a):
    f=open(a,"r")                              #Open file in read mode
    alpha="abcdefghijklmnopqrstuvwxyz"         #Create a string of all alphabets    
    s=""                                       #Cretae an empty string 
    for i in f:
        for j in i:
            if j in alpha:
                s=s+j                          #Using nested for loops to add the letters to the empty string one by one   
    return s                                   #Return the string

'''This takes the file and converts it to a string with all the spaces and other
special characters removed. What remains is only the lower case letters,
retain only the lowercase letters!
'''

def letter_distribution(s):                    
    dict={}                                     #Create an empty dictionary
    alpha="abcdefghijklmnopqrstuvwxyz"          #Cretae a string of alphabets 
    for i in range(len(alpha)):                 #Create a dictionary to store letters mapped to their occurences
        c=s.count(alpha[i])
        dict.update({alpha[i]:c})
    return dict

'''Consider the string s which comprises of only lowercase letters. Count
the number of occurrences of each letter and return a dictionary'''

def substitution_encrypt(s,d):            
    t=""                                        #Create an empty list
    for i in s:                                 #For every letter is s substitue its alternative from the dictionary given
        t+=d[i]
    return t
   

d={'a': 'q', 'b': 'w', 'c': 'e', 'd': 'r', 'e': 't', 'f': 'y', 'g': 'u', 'h': 'i', 'i': 'o', 'j': 'p', 'k': 'l', 'l': 'k', 'm': 'j', 'n': 'h', 'o': 'g', 'p': 'f', 'q': 'd', 'r': 's', 's': 'a', 't': 'z', 'u': 'x', 'v': 'c', 'w': 'v', 'x': 'b', 'y': 'n', 'z': 'm'}

'''encrypt the contents of s by using the dictionary d which comprises of
the substitutions for the 26 letters. Return the resulting string'''

def substitution_decrypt(s,d):
    d1={}                                                               #Create a new empty dictionary 
    for (i,j) in d.items():                                             #Convert the dictionary into tuples and then invert it
        d1[j]=i
    print(substitution_encrypt(s,d1))                                   #Using the inverted dictionary, encrypt the cipher to decrypt it
    
'''decrypt the contents of s by using the dictionary d which comprises of
the substitutions for the 26 letters. Return the resulting string'''

def cryptanalyse_substitution(s):
    d1=letter_distribution(textstrip("ocean60.txt"))                                         #Create a dictionary of the letter distribution of the text file
    d1s=sorted(d1.items(), key= lambda x : x[1], reverse=True)                               #Sort the dictionary in descending order
    d2=letter_distribution(s)                                                                #Create a dictionary of the letter distribution of the string given
    d2s=sorted(d2.items(), key= lambda x : x[1], reverse=True)                               #Sort the dictionary in descending order
    d3={}
    for i in range(26):
        d3[d1s[i][0]]=d2s[i][0]
    return d3                                                                                #Create a new dictionary with the letters of the text file mapped to the letters of the string given
   
'''Given that the string s is given to us and it is known that it was
encrypted using some substitution cipher, predict the d'''

def vigenere_encrypt(s,password):                                                           
    t=""                                                                                       #Create an empty string
    alpha="abcdefghijklmnopqrstuvwxyz"
    for i in range(len(s)):                                                                    #For the password given, shift every letter in s by consecutive indexes of letters in the password
        t+=alpha[(alpha.index(s[i])+alpha.index(password[i%len(password)])+1)%26]
    return t

'''Encrypt the string s based on the password the vigenere cipher way and
return the resulting string'''

def vigenere_decrypt(s,password):
    t=""                                                                                       #Create an empty string
    alpha="abcdefghijklmnopqrstuvwxyz"
    for i in range(len(s)):                                                                    #For the password given, again back shift every letter in s by consecutive indexes of letters
        t+=alpha[(alpha.index(s[i])-alpha.index(password[i%len(password)])-1)%26]
    return t

'''Decrypt the string s based on the password the vigenere cipher way and
return the resulting string'''

def rotate_compare(s,r):
    t=""                                                             #Create empty string
    for i in range(len(s)):                                          #Store the rotated string in the empty string by shifting them
        t+=s[(i+r)%len(s)]                        
    count=0                                                          #Initialise the counter
    for i in range(len(s)):                                          #If words of rotated string collide with original, increment counter
        if t[i]==s[i]:
            count+=1
    return count/len(s)*100                                          #Return the percentage of collisions
    
'''This rotates the string s by r places and compares s(0) with s(r) and
returns the proportion of collisions'''

def cryptanalyse_vigenere_afterlength(s,k):
    pw=""                                                           #Create an empty string                                                           #Create an empty string
    alpha="abcdefghijklmnopqrstuvwxyz"                              #Create a string of all alphabets
    d1s ="etaoinshrdlcumwfgypbvkjxqz"
    for i in range(k):
        t=""
        for j in range(i,len(s),k):
            t+=s[j]                                                 #Create a string of every kth letter in the string given
        d2=letter_distribution(t)                                   #Create a dictionary of the letter distribution of the string given
        d2s=sorted(d2.items(), key= lambda x : x[1], reverse=True)   #Sort the dictionary in descending order
        # print(d2s)
        pw+=alpha[alpha.index(d2s[0][0])-alpha.index(d1s[0])-1]                              #Add the most frequent letter to the password
    return pw

'''Given the string s which is known to be vigenere encrypted with a
password of length k, find out what is the password'''

def cryptanalyse_vigenere_findlength(s):
    i=1                                                              #Initialise the counter
    while(i<len(s)):
        c=rotate_compare(s,i)                                       #Call the rotate compare function
        if c>5:                                                     #If the percentage of collisions is greater than 5, return the counter
            return i
        i+=1                                                       #Else increment the counter

'''Given just the string s, find out the length of the password using which
some text has resulted in the string s. We just need to return the number k'''

def cryptanalyse_vigenere(s):
    k=cryptanalyse_vigenere_findlength(s)                           #Call the function to find the length of the password
    print(k)                                                       #Print the length of the password
    pw=cryptanalyse_vigenere_afterlength(s,k)                     #Call the function to find the password
    print(pw)                                                     #Print the password
    #print(vigenere_decrypt(s,pw))                                #Decrypt the string using the password and print it

cryptanalyse_vigenere(vigenere_encrypt(textstrip("ocean60.txt"),"kulkarni"))              #Call the function to decrypt the string

'''Given the string s cryptanalyse vigenere, output the password as well as
the plaintext'''




