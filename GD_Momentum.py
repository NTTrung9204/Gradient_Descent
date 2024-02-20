import math

class Gradient_Descent:
    def __init__(self, Derivative, Param, Loop = 1000, LR = 0.1, Gamma = 0.9):
        self.Derivative = Derivative
        self.LR = LR
        self.Gamma = Gamma
        self.Param = Param
        self.Loop = Loop
        self.Speed = 0
    
    def GD_Momentum(self):
        for Loop in range(self.Loop):
            self.Speed = self.Gamma * self.Speed + self.LR * self.Derivative(self.Param)
            Temp = self.Param
            self.Param = Temp - self.Speed
            if abs(Temp - self.Param) <= 10**-6:
                print(Loop, self.Param)
                break
        return self.Param

    def GD_NAG(self):
        for Loop in range(self.Loop):
            self.Speed = self.Gamma * self.Speed + self.LR * self.Derivative(self.Param - self.Speed)
            Temp = self.Param
            self.Param = Temp - self.Speed
            if abs(Temp - self.Param) <= 10**-6:
                print(Loop, self.Param)
                break
        return self.Param

    def GD(self):
        for Loop in range(self.Loop):
            Temp = self.Param
            self.Param = self.Param - self.LR * self.Derivative(self.Param)
            if abs(Temp - self.Param) <= 10**-6:
                print(Loop, self.Param)
                break
        return self.Param


def f(x):
    return 2*x + 10 * math.cos(x)

if __name__ == "__main__":
    GDM = Gradient_Descent(f, 8)
    GDM.GD_Momentum()

    GD = Gradient_Descent(f, 8)
    GD.GD()

    GD_NAG = Gradient_Descent(f, 8)
    GD_NAG.GD_NAG()