# Minecraft-Door-Optimization-via-Evolutionary-Algorithms

## Minecraft Server

https://fill-data.papermc.io/v1/objects/cf374f2af9d71dfcc75343f37b722a7abcb091c574131b95e3b13c6fc2cb8fae/paper-1.21.11-69.jar

```sh
/usr/lib/jvm/java-21-openjdk/bin/java -Xmx2G -Xms2G -jar server.jar nogui
```

```sh
python -m src.main --help
```

### Plugins
- https://ci.athion.net/job/FastAsyncWorldEdit/1256/artifact/artifacts/FastAsyncWorldEdit-Paper-2.14.4-SNAPSHOT-1256.jar

---

`DOOR_HEIGHT = 3`
`NGEN = 300`
`mu = lambda = population_size = 100`
`tournsize=3`

<img width="1200" height="1329" alt="image" src="https://github.com/user-attachments/assets/af0ce854-b73f-4618-8918-b65ba0e52538" />

### Showcase: Generated 3x2 Piston Doors

*NOTE: Only one side is displayed, as the problem was simplified to be symmetrical.*

<img width="872" height="738" alt="image" src="https://github.com/user-attachments/assets/e9349b23-2a8a-4e5c-bb6b-5f0454504a0e" />

### Implementation details

#### Individual Representation

Each chunk represents a single individual in the population. Each individual is defined by:
- bounds
- torch position: $(tx, ty, tz) \in \text{BOUNDS}$
- block distribution: a probability distribution for each block type, used for random sampling during mutation

<img width="800" height="600" alt="chunks" src="https://github.com/user-attachments/assets/682ee52a-4253-401f-8984-77a4c99b2cd2" /><br>

#### Hyperplane crossover
<img width="800" height="600" alt="crossover" src="https://github.com/user-attachments/assets/f7f110cd-0b22-44fa-b453-598f90cdc37c" />
