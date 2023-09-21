# rail-segmentation-bisenetV2.5
bisenetV2 plus on railsem 19
This project comes from the rail-segmentation master, thanks to his foundation.

Based on the bisenet he gave, he modified it with a few minor changes and corrections. It ensures that the project can be trained and run smoothly, and some new modules are added to improve the speed of operation (although it does not seem to have done so...).

Optimization: The convolution of detail branches is replaced by deep separable convolution to improve the running speed, and the context module is simplified to improve the running speed.

Run: Load the weight I have trained before, set the image storage address, click run. The current effect is as follows. All objects in the railsem19 database can be recognized, but the results are modest.
weight:链接：https://pan.baidu.com/s/14Q8XUF8Q-lG-QSMWippkHQ?pwd=8ec6  提取码：8ec6
![result](https://github.com/lmcggg/rail-segmentation-bisenetV2.5/assets/78005712/1cda69b8-f1de-4b0b-bb72-5cac44694e63)

Training: Set the database location, weight position, you can train.

it is not very good, can only be used as a simple demonstration, hopefully there will be strong people to optimize.

If you have any questions, please point them out. Thank you.
