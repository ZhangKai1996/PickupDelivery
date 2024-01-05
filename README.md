# PickupDelivery
连续、多目标、多智能体PD场景

## Version Control
- v1.0 (main)
  * 最初的版本；
  * 为了复现经典算法，例如Dijkstra、Astar等，后续的版本需要简化和适配场景；
  * 将单int型坐标换为(int,int)二维坐标；
  * 针对一对一的路径规划，实现了A*、Dijkstra、Best First Search和Bi-A*算法（Test1.py）；
  * 对于突然出现的障碍物，动态调整规划路径，实现了D*算法（Test2.py）；
  * 将网格式的离散空间改为连续空间；
  * 针对连续空间，实现RRT、RRT Connect、Dynamic RRT等算法（Test3.py）；
  * 将离散场景和连续场景合并在一个项目中；
- v1.1
  * 在离散场景中，加入多智能体路径规划，实现SIPP、CBS等中心化算法（Test4.py）；
- v1.2
  * 在连续场景中，加入多智能体路径规划，实现RVO算法；
  * 调试和优化（v1.2.1）；
- v1.3
  * 根据"任务分配-任务整合-路径规划"这一结构调整整个代码的结构（离散场景）；
- v1.4
  * 以连续场景为主，复现分层多智能体强化学习（IDQN+MADDPG）与PD问题的结合；
  * 以MADDPG的代码为主，修改成目标点数量大于智能体数量的环境，并进行初步训练（v1.4.1）；
  * 3个智能体和3个目标点的一对一场景，自主任务分配和最短路径寻找，成功率接近100%（v1.4.3）；
  * 整合前版本备份（v1.4.5）。
- v1.5（改为Pytorch实现，并整合DQN、MADDPG和无人配送连续场景）；
  * 整合代码初步完成，可以训练，需要进一步调试（v1.5.3）；
- v1.6（场景中加入barrier，优化碰撞检测机制，提升代码运行速度）