import random
from os import listdir
from os.path import isfile, join

path = "C:\\Users\\user\\Documents\\renduCVML"


def main():
    files = [path + "\\" + f for f in listdir(path) if isfile(path + "\\" + f) and f.endswith(".txt")]
    lines = []
    dest = open("train2.txt", "w")
    test = open("test2.txt", "w")
    for fileName in files:
        with open(fileName, "r") as f:
            for l in f:
                i = int(l[0])
                if 0 <= i <= 9 and l.count(",") == 256:
                    l = l.replace(" ", "")
                    l = l.replace("[", "")
                    l = l.replace("]", "")
                    if l.endswith("\n"):
                        lines.append(l)
                    else:
                        print("oh noo " + l)
                    print(l, end="")
                else:
                    print("noooo" + l, end="")
                    break
    random.shuffle(lines)
    random.shuffle(lines)
    dest.writelines(lines[:int(len(lines) * 3 / 4)])
    dest.close()
    test.writelines(lines[int(len(lines) * 3 / 4) + 1:])
    test.close()


if __name__ == '__main__':
    main()
