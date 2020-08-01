# Trust Region Competitive Policy Optimization (TRCoPO)
This repository contains code of trust region competitve policy optimisation (TRCoPO) algorithm. The paper for competitive policy gradient can be found [here](https://arxiv.org/abs/2006.10611),
The code for Competitive Policy Gradient (CoPG) algorithm can be found [here](https://github.com/manish-pra/copg). 

## Experiment videos are available [here](https://sites.google.com/view/rl-copo)
## Dependencies
1. Code is tested on python 3.5.2.
2. Only Markov Soccer experiment requires [OpenSpiel library](https://github.com/deepmind/open_spiel), Other experiments can be run directly. 
3. Require [torch.utils.tensorboard](https://pytorch.org/docs/stable/tensorboard.html)

## Repository structure
    .
    ├── notebooks
    │   ├── RockPaperScissors.ipynb
    │   ├── MatchingPennies.ipynb
    ├── game                            # Each game have a saparate folder with this structure
    │   ├── game.py                     
    │   ├── copg_game.py                
    │   ├── gda_game.py
    │   ├── network.py
    ├── copg_optim
    │   ├── copg.py 
    │   ├── critic_functions.py 
    │   ├── utils.py 
    ├── car_racing_simulator
    └── ...
1. [Jupyter notebooks] are the best point to start. It contains demonstrations and results. 
2. Folder [copg_optim] contains optimization code

## How to start ?
Open jupyter notebook and run it to see results.

or

```
git clone "adress"
cd trcopo
cd RockPaperScissors
python3 trcopo_rps.py
cd ..
cd tensorboard
tensordboard --logdir .
```
You can check results in the tensorboard.

## Experiment Demonstration
### &nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  TRGDA vs TRGDA    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                  TRCoPO vs TRCoPO
### ORCA Car Racing
&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;![](https://user-images.githubusercontent.com/37489792/84300401-87121a80-ab52-11ea-995b-3e62ebcddc0b.gif) &nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp; ![](https://user-images.githubusercontent.com/37489792/84300407-88434780-ab52-11ea-8d47-c5f547594617.gif)
### Rock Paper Scissors
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/37489792/84299773-96449880-ab51-11ea-8844-5bc6140ac88c.gif" width="350" height="250">&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; <img src="https://user-images.githubusercontent.com/37489792/84299771-95ac0200-ab51-11ea-8841-a99fd98a0006.gif" width="350" height="250"> 

### Markov Soccer
&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/37489792/84299766-947ad500-ab51-11ea-9dda-1713584abaa0.gif" width="350" height="250">&nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; <img src="https://user-images.githubusercontent.com/37489792/84299762-93e23e80-ab51-11ea-81e6-830e89e2ff10.gif" width="350" height="250"> 

### Matching Pennies
&nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/37489792/84299770-95136b80-ab51-11ea-8f94-b94bda3cb7ac.gif" width="350" height="250">&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;<img src="https://user-images.githubusercontent.com/37489792/84299768-947ad500-ab51-11ea-865c-5aaa2e98d18e.gif" width="350" height="250"> 


