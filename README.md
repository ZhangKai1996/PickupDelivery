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
