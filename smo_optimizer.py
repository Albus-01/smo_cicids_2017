import time
import random
import numpy
import math
import torch

def objective_function(weights, model, loss_fn, X_train, y_train, device):
    """
    Calculates the loss for a given set of weights.
    This function is minimized by the SMO algorithm.
    """
    # Deconstruct the flattened weights array back into model layers
    layer1_weights = torch.from_numpy(weights[0:2496].reshape(32, 78)).float().to(device)
    layer1_bias = torch.from_numpy(weights[2496:2528].reshape(32,)).float().to(device)
    layer2_weights = torch.from_numpy(weights[2528:3552].reshape(32, 32)).float().to(device)
    layer2_bias = torch.from_numpy(weights[3552:3584].reshape(32,)).float().to(device)
    layer3_weights = torch.from_numpy(weights[3584:3616].reshape(1, 32)).float().to(device)
    layer3_bias = torch.from_numpy(weights[3616:3617].reshape(1,)).float().to(device)

    # Assign the new weights to the model
    with torch.no_grad():
        model.layer_stack[0].weight.data = layer1_weights
        model.layer_stack[0].bias.data = layer1_bias
        model.layer_stack[2].weight.data = layer2_weights
        model.layer_stack[2].bias.data = layer2_bias
        model.layer_stack[4].weight.data = layer3_weights
        model.layer_stack[4].bias.data = layer3_bias

    model.eval()
    with torch.no_grad():
        y_logits = model(X_train).squeeze()
        loss = loss_fn(y_logits, y_train)
    
    return loss.item()

class SMO():
    def __init__(self, objf, lb, ub, dim, pop_size, acc_err, iters):
        self.PopSize = pop_size
        self.dim = dim
        self.acc_err = acc_err
        self.lb = lb
        self.ub = ub
        self.objf = objf
        self.pos = numpy.zeros((pop_size, dim))
        self.fun_val = numpy.zeros(pop_size)
        self.fitness = numpy.zeros(pop_size)
        self.gpoint = numpy.zeros((pop_size, 2))
        self.prob = numpy.zeros(pop_size)
        self.LocalLimit = dim * pop_size
        self.GlobalLimit = pop_size
        self.fit = numpy.zeros(pop_size)
        self.MinCost = numpy.zeros(iters)
        self.Bestpos = numpy.zeros(dim)
        self.group = 10
        self.func_eval = 0
        self.part = 1
        self.max_part = 5
        self.cr = 0.1

        self.GlobalMin = float('inf')
        self.GlobalLeaderPosition = numpy.zeros(dim)
        self.GlobalLimitCount = 0
        S_max = int(self.PopSize / 2)
        self.LocalMin = numpy.zeros(S_max)
        self.LocalLeaderPosition = numpy.zeros((S_max, self.dim))
        self.LocalLimitCount = numpy.zeros(S_max)

    def CalculateFitness(self, fun1):
      return (1 / (fun1 + 1)) if fun1 >= 0 else (1 + math.fabs(fun1))

    def initialize(self):
        for i in range(self.PopSize):
            self.pos[i,:] = numpy.random.uniform(0, 1, self.dim) * (self.ub - self.lb) + self.lb
            self.pos[i,:] = numpy.clip(self.pos[i,:], self.lb, self.ub)
            self.fun_val[i] = self.objf(self.pos[i,:])
            self.func_eval += 1
            self.fitness[i] = self.CalculateFitness(self.fun_val[i])

        self.GlobalMin = self.fun_val[0]
        self.GlobalLeaderPosition = self.pos[0,:]
        self.GlobalLimitCount = 0

        for k in range(self.group):
            self.LocalMin[k] = self.fun_val[int(self.gpoint[k,0])]
            self.LocalLimitCount[k] = 0
            self.LocalLeaderPosition[k,:] = self.pos[int(self.gpoint[k,0]),:]

    def CalculateProbabilities(self):
        maxfit = numpy.max(self.fitness)
        self.prob = (0.9 * (self.fitness / maxfit)) + 0.1

    def create_group(self):
        g = 0
        lo = 0
        while(lo < self.PopSize):
            hi = lo + int(self.PopSize / self.part)
            self.gpoint[g, 0] = lo
            self.gpoint[g, 1] = hi
            if ((self.PopSize - hi) < (int(self.PopSize / self.part))):
                self.gpoint[g, 1] = (self.PopSize - 1)
            g += 1
            lo = hi + 1
        self.group = g

    def LocalLearning(self):
        OldMin = self.LocalMin.copy()
        for k in range(self.group):
            group_indices = range(int(self.gpoint[k, 0]), int(self.gpoint[k, 1]) + 1)
            if not group_indices: continue
            
            group_fun_val = self.fun_val[group_indices]
            min_idx_local = numpy.argmin(group_fun_val)
            min_val_local = group_fun_val[min_idx_local]
            
            if min_val_local < self.LocalMin[k]:
                self.LocalMin[k] = min_val_local
                self.LocalLeaderPosition[k,:] = self.pos[group_indices[min_idx_local],:]

        for k in range(self.group):
            if math.fabs(OldMin[k] - self.LocalMin[k]) < self.acc_err:
                self.LocalLimitCount[k] += 1
            else:
                self.LocalLimitCount[k] = 0

    def GlobalLearning(self):
        G_trial = self.GlobalMin
        min_val_global = numpy.min(self.fun_val)
        if min_val_global < self.GlobalMin:
            self.GlobalMin = min_val_global
            self.GlobalLeaderPosition = self.pos[numpy.argmin(self.fun_val),:]

        if math.fabs(G_trial - self.GlobalMin) < self.acc_err:
            self.GlobalLimitCount += 1
        else:
            self.GlobalLimitCount = 0

    def LocalLeaderPhase(self, k):
        lo, hi = int(self.gpoint[k, 0]), int(self.gpoint[k, 1])
        if lo >= hi: return

        for i in range(lo, hi + 1):
            PopRand = i
            while PopRand == i:
                PopRand = random.randint(lo, hi)
            
            new_position = self.pos[i,:].copy()
            for j in range(self.dim):
                if random.random() >= self.cr:
                    new_position[j] += (self.LocalLeaderPosition[k, j] - self.pos[i, j]) * random.random() + \
                                       (self.pos[PopRand, j] - self.pos[i, j]) * (random.random() - 0.5) * 2

            new_position = numpy.clip(new_position, self.lb, self.ub)
            ObjValSol = self.objf(new_position)
            self.func_eval += 1
            FitnessSol = self.CalculateFitness(ObjValSol)
            if FitnessSol > self.fitness[i]:
                self.pos[i,:] = new_position
                self.fun_val[i] = ObjValSol
                self.fitness[i] = FitnessSol

    def GlobalLeaderPhase(self, k):
        lo, hi = int(self.gpoint[k, 0]), int(self.gpoint[k, 1])
        if lo >= hi: return
        
        i = lo
        count = 0
        while count < (hi - lo + 1):
            if random.random() < self.prob[i]:
                PopRand = i
                while PopRand == i:
                    PopRand = random.randint(lo, hi)
                
                param2change = random.randint(0, self.dim - 1)
                new_position = self.pos[i, :].copy()
                new_position[param2change] += (self.GlobalLeaderPosition[param2change] - self.pos[i, param2change]) * random.random() + \
                                              (self.pos[PopRand, param2change] - self.pos[i, param2change]) * (random.random() - 0.5) * 2
                
                new_position = numpy.clip(new_position, self.lb, self.ub)
                ObjValSol = self.objf(new_position)
                self.func_eval += 1
                FitnessSol = self.CalculateFitness(ObjValSol)
                if FitnessSol > self.fitness[i]:
                    self.pos[i,:] = new_position
                    self.fun_val[i] = ObjValSol
                    self.fitness[i] = FitnessSol
            i = i + 1 if i < hi else lo
            count += 1

    def GlobalLeaderDecision(self):
        if self.GlobalLimitCount > self.GlobalLimit:
            self.GlobalLimitCount = 0
            self.part = self.part + 1 if self.part < self.max_part else 1
            self.create_group()
            self.LocalLearning()

    def LocalLeaderDecision(self):
        for k in range(self.group):
            if self.LocalLimitCount[k] > self.LocalLimit:
                for i in range(int(self.gpoint[k, 0]), int(self.gpoint[k, 1]) + 1):
                    for j in range(self.dim):
                        if random.random() >= self.cr:
                           self.pos[i, j] = numpy.random.uniform(0, 1) * (self.ub[j] - self.lb[j]) + self.lb[j]
                        else:
                            self.pos[i, j] += (self.GlobalLeaderPosition[j] - self.pos[i, j]) * random.random() + \
                                              (self.pos[i, j] - self.LocalLeaderPosition[k, j]) * random.random()
                    
                    self.pos[i, :] = numpy.clip(self.pos[i,:], self.lb, self.ub)
                    self.fun_val[i] = self.objf(self.pos[i, :])
                    self.func_eval += 1
                    self.fitness[i] = self.CalculateFitness(self.fun_val[i])
                self.LocalLimitCount[k] = 0

def run_smo(objf, lb, ub, dim, pop_size, iters, acc_err, obj_val=0):
    smo = SMO(objf, lb, ub, dim, pop_size, acc_err, iters)
    smo.initialize()
    smo.GlobalLearning()
    smo.LocalLearning()
    smo.create_group()

    for l in range(iters):
        for k in range(smo.group):
            smo.LocalLeaderPhase(k)

        smo.CalculateProbabilities()

        for k in range(smo.group):
            smo.GlobalLeaderPhase(k)

        smo.GlobalLearning()
        smo.LocalLearning()
        smo.LocalLeaderDecision()
        smo.GlobalLeaderDecision()
        
        smo.cr += (0.4 / iters)
        smo.MinCost[l] = smo.GlobalMin

        if (l % 10 == 0):
               print(f'Iteration {l+1}: Best Fitness = {smo.GlobalMin}')

        if math.fabs(smo.GlobalMin - obj_val) <= smo.acc_err:
            break
            
    return smo.GlobalLeaderPosition