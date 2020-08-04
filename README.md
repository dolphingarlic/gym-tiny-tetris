# Tiny Tetris

An OpenAI `gym` env for the game Tiny Tetris from the 2012 Baltic Olympiad in Informatics.

[Original problem statement](http://www.boi2012.lv/data/day2/eng/tiny.pdf).

## Usage

To install on Google Colab:

```bash
%%capture
!rm -r tiny-tetris
!git clone https://github.com/dolphingarlic/tiny-tetris.git
!pip install -e tiny-tetris

!pip uninstall -y tensorflow
!pip install tensorflow==1.14.0
!pip install stable-baselines
```

To train and predict:

```py
import gym

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

from gym_tiny_tetris.envs.tiny_tetris_env import TinyTetrisEnv

env = TinyTetrisEnv(5)
env = DummyVecEnv([lambda: env])

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=200000)

obs = env.reset()
with open('tiny.o5', 'w') as fout:
  for i in range(2000):
      action, _states = model.predict(obs)
      obs, rewards, done, info = env.step(action)
      env.render()
      if done:
        break
      fout.write(f'{action[0] + 1}\n')

env.close()
```
