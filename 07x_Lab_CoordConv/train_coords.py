"""Click-to-run wrapper: same as train.py but with coordinate channels ON.
Double-click this file (instead of train.py) for the CoordConv treatment run.
"""
import os, sys, runpy
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ['train.py', '--coords']
runpy.run_path('train.py', run_name='__main__')
