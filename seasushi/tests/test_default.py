#%%
import unittest
from seasushi.tools import Sushi

class TestSushi(unittest.TestCase):
    def test_sushi_creation(self):
        sushi_instance = Sushi('Salmon', 'Nigiri')
        sushi = sushi_instance.make_sushi()
        self.assertEqual(sushi.type, 'Nigiri')
        self.assertEqual(sushi.fish, 'Salmon')
        return print(sushi)
if __name__ == '__main__':
    unittest.main()
# %%
