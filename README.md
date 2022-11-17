# Maratona de Programação Paralela da ERAD/RS 2021

Esta página descreve o desafio proposta na Maratona de Programação Paralela da ERAD/RS 2021. A maratona deste ano conta com o apoio da [Dell](https://www.dell.com/pt-br) e [NVIDIA](https://www.nvidia.com/pt-br/).

Placar: https://mpp-eradrs.github.io/

Boa sorte a todos e bom desafio!

# Desafio proposto - miniCFD

O desafio deste ano consiste em encontrar soluções de HPC para um problema de dinâmica de fluidos. A simulação consiste na inserção de fluxo no centro de um stencil neutro. O código é semelhante ao que se encontra na grade maioria das aplicações de dinâmica de fluidos. 

O código sequencial umas 500 linhas de código sendo que 200 são partes importantes para otimização. 
As dimensões são `x` e `z` sendo `nx_cfd` e `nz_cfd` o tamanho global das dimensões e `nnx` e `nnz` no caso do tamanho local (MPI). 

Algumas das principais vetores do código são:
- `state` - o estado atual da simulação, e o único que persiste entre iterações;
- `state_tmp`- cópia temporário usada na integração Runge-Kutta;
- `flux` - estado nas bordas para as dimensões `x` e `z`;
- `tend` - tendência dos deltas `q/t` onde `q` é o vetor de estados e `t` tempo.


O laço principal está abaixo. A função `do_timestep` executa a iteração.
```c++
  ////////////////////////////////////////////////////
  // MAIN TIME STEP LOOP
  ////////////////////////////////////////////////////
  auto c_start = std::clock();
  while (etime < sim_time) {
    //If the time step leads to exceeding the simulation time, shorten it for the last step
    if (etime + dt > sim_time) { dt = sim_time - etime; }
    //Perform a single time step
    do_timestep(state,state_tmp,flux,tend,dt);
    //Update the elapsed time and output counter
    etime = etime + dt;
    output_counter = output_counter + dt;
    //If it's time for output, reset the counter, and do output
    if (output_counter >= output_freq) {
      output_counter = output_counter - output_freq;
      //Inform the user
      if (masterproc) { printf( "Elapsed Time: %lf / %lf\n", etime , sim_time ); }  
    }
  }
  auto c_end = std::clock();
```

Alguns trechos do código tem o comentário abaixo. Esses são algumas das dicas que o código fornece.
```c++
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
```

## Arquivos e execução

O desafio consiste nos arquivos abaixo:
- `miniCFD_serial.cpp` - código serial do problema;
- `env.sh` - arquivo com configurações de módulos de software e variáveis de ambiente. Pode ser alterado;
- `build.sh` - compila a aplicação com compilador e opções necessárias. Pode ser alterado;
- `run.sh` - roda o problema de acordo. Pode ser alterado.

**Não modifique os parâmetros de entrada do problema** que estão no arquivo `build.sh`:
- `_D_NX` - tamanho na dimensão `x` (padrão: 800);
- `_D_NZ`- tamanho na dimensão `z` (padrão: 400)
- `_SIM_TIME` - tempo em segundos de simulação (padrão: 400)
- `_D_OUT_FREQ` - frequência de saída
- `_D_IN_CONFIG` - configuração da simulação (padrão: `CONFIG_IN_TEST1`)


## Critérios de avaliação

A principal métrica será o **Speedup** com relação ao melhor tempo da versão sequêncial da aplicação.

Cada avaliação de resultado terá a sequência de comandos abaixo:
```sh
source env.sh
./build.sh
time -p ./run.sh
```

O tempo será medido diversas vezes por meio do comando `time` usando o tempo real de execução. O número de repetições realizadas será:
- 10x repetições durante a maratona
- 30x repetições para a avaliação final

Será considerado empate quando **o speedup é igual** considerando três casas decimais. Os critérios de desempate aplicados serão na ordem abaixo:
- Média do tempo de execução;
- Mediana do tempo de execução;
- Menor desvio padrão.

## Verificação do resultado numérico

As duas últimas linhas de saída da simulação são:
- `d_mass: 7.848194e-13`
- `d_te: 9.145021e-05`

Os valores podem mudar com as alterações no programas. Certifique-se de que os valores não mudem de forma considerável.

## Regras

Caso a equipe não uma das regras abaixo será automaticamente desclassificado da avaliação final.

- Deve utilizar ao menos uma forma de paralelização do código. Somente otimizações do código sequencial não serão aceitos.
- Não modifique a precisão, tamanho ou lógica do problema. Os julgadores podem desclassficiar uma equipe por alterações na lógica e/ou resultado do problema tais como precisão, simplificação de cálculos, etc.
- Confira se a solução da simulação está correta. 
- Não será permitido plágio.

## Ferramentas sugeridas

A lista abaixo é uma sugestão e não restringe as escolhas:
- OpenMP
- CUDA
- MPI
- OpenACC
- OpenCL

# Plataforma de testes

As equipes tem a disposição 10 nós do [Dell HPC & AI Innovation Lab](https://www.delltechnologies.com/pt-br/solutions/high-performance-computing/HPC-AI-Innovation-Lab.htm) equipados com:
- 5 x PowerEdge C4140 (Intel Xeon Gold 6148, 2.40GHz, 384 GB, 2666 MHz memory, 120 GB SSDs, 4x V100-32GB GPU)
- 5 x PowerEdge C4140 (Intel Xeon Gold 6148, 2.40GHz, 384 GB, 2666 MHz memory, 120 GB SSDs, 4x V100-16GB GPU)

As máquina do cluster disponíveis para a maratona são:
- gpu[003-005], gpu017, gpu024 - 4x V100 32GB GPU
- gpu[018-020], gpu023, gpu025 - 4x V100 16GB GPU

## Acesso

Utilize o usuário e senha enviados a cada equipe por email. Em seguida, acesse o cluster `Rattler` por SSH:
```
$ ssh rattler
```
Você estará em um dos nós de acesso ao cluster (=rlogin01= ou =rlogin02=). 

Acesse qualquer um deles por SSH:
```
$ ssh gpu003
```

## Software de desenvolvimento

Bibliotecas e compiladores estão disponíveis por meio dos pacotes =module= dentro de cada nó:
```
$ module avail
```

Por exemplo, o NVIDIA HPC está disponível no módulo:
```
$ module load nvidia/nvhpc/21.3
```

# Submissão do problema 


As submissões serão por meio de um repositório GitHub Classroom.  O sistema de avaliação irá atualizar os resultados baseados nos repositório criados.

Cada equipe terá seu repositório com acesso restrito. 
Este repositório será ligado ao usuário GitHub da pessoa que criar o respositório podendo adicionar colaboradores da mesma equipe em seguida. O repositório pode ser criado através do link enviado por email.

Os passos para criar o seu repositório com a sua equipe são:
- Entre no link acima
- Escolha o nome do sua equipe na lista para criar o respositório
- Siga as instruções
- A primeira tela de seu novo repositório já permite que mais usuários sejam convidados no botão *Add teams and collaborators*.

Se tiver dificuldades para adicionar um usuário em sua equipe acesse:
- https://docs.github.com/en/github/setting-up-and-managing-your-github-user-account/inviting-collaborators-to-a-personal-repository

Um resumo sobre comandos Git pode ser encontrado no site abaixo. Dúvidas podem ser enviadas a organização no canal Discord.
- https://training.github.com/downloads/github-git-cheat-sheet/


# Dicas de análise de código

Sugerimos duas ferramentas para avaliação do código paralelo:
- [nvprof](https://docs.nvidia.com/cuda/profiler-users-guide/index.html)
- [Nsight](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

O `nvprof` analisa e gera rastros de chamadas CUDA do programa e o programador pode marcar regiões de avaliação do código. Dado um programa com chamadas CUDA:
```
$ nvprof ./application
```

O comando `nsys` permite gerar rastros de chamadas CPU e GPU tendo suporte a uma gama maior de ferramentas. O primeiro passo é gerar o rastro da aplição com o comando:
```
$ nsys profile -t cuda,openmp --stats=true --force-overwrite true -o app_r1 ./application
```

O comando acima já imprime as estatísticas de execução. Para imprimir novamente:
```
nsys stats app_r1.qdrep
``` 

# Organização

A organização da MPP ERAD/RS 2021 ficou com os professores:
- Dalvan Griebler (PUCRS/SETREM)
- João Vicente Ferreira Lima (UFSM)
