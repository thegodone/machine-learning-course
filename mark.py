path = 'reviews/'
pred_path = path + 'exp2.validation_review.csv'
pred_file = open(pred_path, 'r')
full_data = []
pred_data = []
num_lines = 0
ans = []
for line in pred_file.readlines():
    num_lines += 1
    if num_lines == 1:
        continue
    print(line[:-1])
    a = raw_input('Tag is: ')
    ans.append(a)

    t = open('mark.csv', 'a')
    t.write('%d,%s\n' % (num_lines, a))
    t.close()
