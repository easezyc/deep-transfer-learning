class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Stoper():

    def __init__(self, early_step):
        self.max = 0
        self.cur = 0
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1

    def judgesave(self):
        if self.cur > self.max:
            return True
        else:
            return False

    def judge(self):
        if self.cur > self.max:
            self.max = self.cur
            self.maxindex = self.curindex
        if self.curindex - self.maxindex >= self.early_step:
            return True
        else:
            return False

    def showfinal(self):
        result = "AUC {}\n".format(self.max)
        print(result)
        return self.max
