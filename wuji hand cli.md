conda activate wuji_hw
unset PYTHONPATH

lsusb | grep -Ei '0483|wuji'

--------

import wujihandpy

hand = wujihandpy.Hand()

q = hand.read_joint_actual_position()
print(q)
