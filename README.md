# 微电机人形机器人与复合肌体仿真器

这个原型模拟“数千到上万个 mm 尺度微电机单元驱动的人形机器人”，并保留一个平面复合肌体作为对照实验台。它不是把每个电机当成孤立伺服器，而是把高密度电机阵列抽象为分布式肌群：低维顶层运动命令生成关节目标，自适应控制器根据误差、热状态和保护状态调整增益，再把驱动展开到附着在 206 块骨头上的 motor pixel。

## 运行

```bash
cd motor-muscle-sim
npm run smoke
npm start
```

打开 `http://127.0.0.1:4173`。

## MuJoCo 研究后端

`research/` 包含独立的 Python 3.11/MuJoCo 验证环境：19 个动力学刚体、
16 个受控关节和最多 20,000 个具有独立电流、温度、制造偏差与故障状态的
微电机。它提供教师数据生成、连续神经修正策略训练、三控制器对照实验、报告
和视频输出。安装及完整实验命令见
[`research/README.md`](research/README.md)。

## 建模层次

- **运动模式层**：行波步态、下蹲卷曲、双臂抓取、躯干扭转、蠕动步态、局部平衡修正。
- **206 骨骼图谱层**：颅骨、听小骨、舌骨、椎骨、肋骨、胸骨、肩带、上肢、骨盆、下肢、手骨和足骨全部进入 `ANATOMY_206`，启动时硬性校验数量。
- **人形骨架层**：脊柱、颈部、肩、肘、髋、膝、踝关节有角度、角速度、惯量、阻尼、限位和被动刚度。
- **分布式肌群层**：头、躯干、骨盆、上臂、前臂、手、大腿、小腿、足部分别铺设二维微电机阵列。
- **骨级执行器附着层**：每个 motor pixel 有 `boneIndex`，每块骨头都有非零 `boneMotorCounts`，浏览器指标显示 `206/206`。
- **自适应顶层控制层**：顶层模式只表达“走、卷曲、抓取、扭转、局部平衡”等意图，控制器通过 `adaptiveGain` 和 `adaptiveBias` 把误差反馈转换为局部执行器驱动。
- **神经场层**：每个单元有激活状态，支持局部波动、关节反射反馈和热抑制。
- **微电机层**：每个单元有电流、电压、电阻、力矩增益、制造偏差、反电动势和焦耳热。
- **复合肌体平面对照层**：高度场通过拉普拉斯弹性耦合，模拟柔性基体中的相邻力传递。
- **保护层**：温度与局部应变触发降额，避免热失控或过度变形。

## 状态变量

人形机器人每个 motor pixel 使用 typed arrays 存储：

```text
localX, localY, segmentIndex, jointIndex, motorSign, momentArm,
activation, current, temperature, strain, deformation,
torqueGain, resistance, protection, boneIndex, boneMotorCounts,
adaptiveGain, adaptiveBias, trackingError
```

平面复合肌体每个 motor pixel 使用 typed arrays 存储：

```text
height, velocity, activation, current, temperature, strain,
voltage, torqueGain, resistance, mass, protection
```

这让浏览器版本默认运行约 6,000 个电机的人形机器人，并能切换到约 12,000 或 19,000 个电机的高密度版本。相同数据布局也方便后续迁移到 WebGPU、NVIDIA Warp、Taichi 或 C++ 后端。

## 人形机器人计算路径

```text
顶层运动模式 -> 关节目标角
关节误差 + 热/保护压力 -> 自适应增益和偏置
自适应控制输出 -> 206 骨上的 motor pixel 激活
局部激活 -> 电流/热/保护 -> 骨附着执行器力矩贡献
成千上万个力矩贡献 -> 关节净力矩
关节净力矩 -> 人形姿态积分 -> 下一帧可视化
```

人形可视化不是预制动画：画面中的每个彩色点都是一个微电机状态采样，位置随骨架正向运动学实时变化。

## 206 骨骼覆盖

`src/anatomy206.js` 生成并校验标准成人骨骼数量：

```text
轴骨 80: 颅骨 22, 听小骨 6, 舌骨 1, 椎骨 26, 胸骨 1, 肋骨 24
附肢骨 126: 肩带 4, 上肢/手 60, 骨盆 2, 下肢/足 60
总计 206
```

每块骨头都带有 `segment`、`joint`、`weight` 和 `side`，仿真初始化时按身体段把微电机分配到骨头上。`npm run smoke` 会断言 `coveredBones === 206`，并检查所有 `boneMotorCounts` 都大于 0。

## 参考的先进仿真思想

- 数据并行物理核：类似 NVIDIA Warp 和 DiffTaichi 的“数组状态 + kernel step”结构。
- 执行器状态模型：参考 MuJoCo 的 actuator activation/control/force 分层思想。
- 大规模耦合动力学：参考 Chrono 对大量机械单元、柔性结构和多体系统的处理思路。
- 形态计算：把柔性基体的耦合和滞后视为控制系统的一部分，而不是纯粹的扰动。

## 后续可扩展方向

- WebGPU 计算 shader，把每个单元更新迁移到 GPU。
- 加入真实电机 CAD/磁路参数表，生成不同制造工艺的阵列配置。
- 加入逆设计或可微优化，让目标形变自动反推协同模式。
- 加入多层结构：表层触觉、电机层、柔性骨架层、冷却层。
- 加入接触与地面反力，把当前“空中步态/姿态演示”推进到可验证的动力学行走。
