import unittest
from inferencer.YAMnet import YAMnet

class TestStringMethods(unittest.TestCase):

    def test_inferencer(self):
        inferencer = YAMnet()
        result = inferencer.run_inferencer('test/97.5DbAudio.wav')
        assert 'audio_filename' in result


if __name__ == '__main__':
    unittest.main()