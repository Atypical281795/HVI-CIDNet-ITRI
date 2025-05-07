# test_options.py
import sys
sys.path.insert(0, './data')
from options import option

parser = option()
print("Available arguments:")
for action in parser._actions:
    if hasattr(action, 'option_strings'):
        print(f"  {action.option_strings}")

try:
    opt = parser.parse_args(['--CEC'])
    print("Success! CEC value:", opt.CEC)
except Exception as e:
    print("Error:", e)