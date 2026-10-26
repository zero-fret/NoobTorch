"""Click-to-run wrapper: test the CoordConv treatment weights (model_coords_True.pth).
"""
import os, sys, runpy
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ['test.py', '--coords']
runpy.run_path('test.py', run_name='__main__')
