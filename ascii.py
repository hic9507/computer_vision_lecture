# # b = '◯'
# # c = b.encode('utf-8').decode('\u25ef')
# # answkduf = answkduf.encode('utf-8')
# # aaa = answkduf+c

# # d = c.decode('utf-8')

# string = '\x08'
# string1 = string.encode('utf-8')#.decode('\x08')
# string2 = string.decode('utf-8')
# print(string1)
# print(string2)

def unicode_test(value):
    import unicodedata
    name = unicodedata.name(value)
    value2 = unicodedata.lookup(name)
    print('value = "%s", name = "%s", value2 = "%s"' % (value, name, value2))

# print(unicode_test('A'))
# print(unicode_test('$'))
# print(unicode_test('\u00a2'))
# print(unicode_test('\u20ac'))
# print(unicode_test('\u2603'))

print('-'*100)
a0 = '\x8b'
a1 = a0.encode('utf-16')
a2 = a0.encode('utf-8')
# a3 = a0.encode('cp949')
a4 = a0.encode('latin1')
print(a1)
print(a2)
# print(a3)
print(a4)
print('-'*100)

snowman = '\x8b'
ds = snowman.encode('utf-16')
print(len(ds))
print(ds)
# a = snowman.decode('utf-8')