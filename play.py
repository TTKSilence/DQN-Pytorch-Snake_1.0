# for snake(snaky)
import snaky as game
import cv2

# for tensor
import numpy as np
import tensorflow as tf
import random
from collections import deque

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        # 8 x 8 x 4 with 32 Filters ,Stride 4 -> Output 21 x 21 x 32 -> max_pool 11 x 11 x 32
        self.conv1=nn.Conv2d(4,32,kernel_size=8,stride=4,padding = 2)
        self.conv1.weight.data.normal_(0,0.1)
        self.mp1=nn.MaxPool2d(2,stride=2,padding=1)  #2x2x32 stride=2
        # 4 x 4 x 32 with 64 Filters ,Stride 2 -> Output 6 x 6 x 64
        self.conv2=nn.Conv2d(32,64,kernel_size=4,stride=2,padding=2)
        self.conv2.weight.data.normal_(0,0.1)
        # 3 x 3 x 64 with 64 Filters,Stride 1 -> Output 6 x 6 x 64
        self.conv3=nn.Conv2d(64,64,kernel_size=3,stride=1,padding = 1)
        self.conv3.weight.data.normal_(0,0.1)

        self.fc1=nn.Linear(2304,4)
        self.fc1.weight.data.normal_(0,0.1)
    
    def forward(self,x):
            
        x=F.relu(self.conv1(x))
        x=self.mp1((x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))     
        conv3_to_reshaped=torch.reshape(x,[-1,2304])
        output=self.fc1(conv3_to_reshaped)
        return output  #action's value

# Based on NIPS 2013
class DQN:

    def __init__(self):
        
        self.memory = deque()     # initialize memory
        self.eval_net,self.target_net=Net(),Net()
        self.optimize=torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss=nn.MSELoss()

        #Net参数初始化
        self.round = 0  #iteration counter
        #Hyperparameters
        self.epoch=0   #memory counter
        self.observe=10000
        self.epsilon=1 #INIT_EPSILON
        self.finep =0.9 # FIN_EPSILON
        self.actions=4 #ACTIONS
        
    def get_action(self):
        self.s_t = torch.unsqueeze(torch.FloatTensor(self.s_t), 0)
        action = np.zeros(self.actions)
        if np.random.uniform() < self.epsilon:    #  choose greedy way
            idx = np.random.randint(0, N_ACTIONS)
            action[idx]=1
            #action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

        else:   # random
            actions_value = self.eval_net.forward(self.s_t)
            print(actions_value.shape)
            action = torch.max(actions_value, 1)[1].data.numpy()
            
        # change episilon ——训练期
        if self.epsilon > self.finep and self.epoch > self.observe:
            self.epsilon -= (1 - self.finep) / 500000 
        return action

    def store_transition(self,s1,a,r,done):
        #transition = np.hstack((s, a, r, done))
        print(self.s_t.shape)
        print('ok')
        print(s1.shape)
        tmp = np.append(self.s_t[1:,:,:], s1, axis = 0)
        self.memory.append((self.s_t, a, r, tmp, done))
        # replace the old memory with new memory
        if len(self.memory) > MEMORY_CAPACITY:
            self.memory.popleft()
        if self.epoch > self.observe:
            self.learn()
        self.s_t=tmp
        self.epoch += 1

        return self.epoch   
    
    def initState(self, state):
        self.s_t = np.stack((state, state, state, state), axis=2)

class agent:
    def screen_handle(self, screen):
        procs_screen = cv2.cvtColor(cv2.resize(screen, (84, 84)), cv2.COLOR_BGR2GRAY)
        dummy, bin_screen = cv2.threshold(procs_screen, 1, 255, cv2.THRESH_BINARY)
        bin_screen = np.reshape(bin_screen, (84, 84, 1))
        return bin_screen
        
    def run(self):
        # initialize
        # discount factor 0.99
        ag = DQN(0.99, 0, 0.001, 0.001, 50000, 32, 4)
        g = game.gameState()
        a_0 = np.array([1, 0, 0, 0])
        s_0, r_0, d = g.frameStep(a_0)
        s_0 = cv2.cvtColor(cv2.resize(s_0, (84, 84)), cv2.COLOR_BGR2GRAY)
        _, s_0 = cv2.threshold(s_0, 1, 255, cv2.THRESH_BINARY)
        ag.initState(s_0)
        while True:
            a = ag.getAction()
            s_t1, r, done = g.frameStep(a)
            s_t1 = self.screen_handle(s_t1)
            ts, qv = ag.addReplay(s_t1, a, r, done)
            # for Summary
            if done == True:
                sc, ep = g.retScore()
                print(ts,",",qv,",",ep, ",", sc)
            else:
                print(ts,",",qv,",,")

def main():
    run_agent = agent()
    run_agent.run()

if __name__ == '__main__':
    main()