import pygame
import sys
import neat
import os
import random
import logging
import pickle
from typing import Optional
from io import StringIO

# ─────────────────────────────────────────────
#  Constantes do Jogo
# ─────────────────────────────────────────────
LARGURA_TELA = 800
ALTURA_TELA = 600
TAMANHO_CELULA = 50
QUADROS_POR_SEGUNDO = 60

FAIXAS_MUNDO = 80          # número total de faixas no mundo
FAIXAS_VISIVEIS = ALTURA_TELA // TAMANHO_CELULA

# Faixas de grama a cada N faixas de estrada
BLOCO_ASFALTO = 3   # faixas de asfalto por bloco
BLOCO_GRAMA = 1     # faixas de grama entre blocos

MAX_TIQUES = 9000   # tiques máximos por geração

# Balanceamento de tráfego
VELOCIDADE_MINIMA_BASE = 1.2
VELOCIDADE_MAXIMA_BASE = 2.6
TETO_VELOCIDADE_MAXIMA = 4.0

# ─────────────────────────────────────────────
#  Cores
# ─────────────────────────────────────────────
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
GRAMA_CLARA = (86, 170, 73)
GRAMA_ESCURA = (62, 137, 52)
ASFALTO_CLARO = (100, 100, 100)
ASFALTO_ESCURO = (80, 80, 80)
AMARELO = (255, 215, 0)
VERDE_SAPO = (50, 200, 50)

CORES_CARROS = [
    (220, 50, 30),
    (30, 100, 220),
    (230, 180, 0),
    (200, 80, 0),
    (140, 40, 200),
]


# ─────────────────────────────────────────────
#  Persistência do Melhor Genoma
# ─────────────────────────────────────────────
def _caminho_melhor_genoma() -> str:
    return os.path.join(os.path.dirname(__file__), "best_genome.pkl")


def carregar_melhor_genoma() -> Optional[neat.DefaultGenome]:
    caminho = _caminho_melhor_genoma()
    if not os.path.exists(caminho):
        return None
    try:
        with open(caminho, "rb") as arquivo:
            return pickle.load(arquivo)
    except Exception:
        return None


def salvar_melhor_genoma(genoma: neat.DefaultGenome):
    caminho = _caminho_melhor_genoma()
    with open(caminho, "wb") as arquivo:
        pickle.dump(genoma, arquivo)


# ─────────────────────────────────────────────
#  Configuração de Logging
# ─────────────────────────────────────────────
def configurar_log(arquivo_log: str = "training_log.txt"):
    """Configura o logging para salvar em arquivo txt."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(arquivo_log, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Classe Carro
# ─────────────────────────────────────────────
class Carro:
    LARGURA = TAMANHO_CELULA * 2
    ALTURA = TAMANHO_CELULA - 8

    def __init__(self, faixa_mundo_y: int, velocidade: float, direcao: int, x: Optional[float] = None):
        self.faixa_mundo_y = faixa_mundo_y
        self.velocidade = velocidade
        self.direcao = direcao   # +1 = direita, -1 = esquerda
        self.cor = random.choice(CORES_CARROS)

        if x is None:
            self.x = -self.LARGURA if direcao == 1 else LARGURA_TELA + self.LARGURA
        else:
            self.x = float(x)

    def y_tela(self, y_camera: int) -> int:
        return self.faixa_mundo_y - y_camera + 4

    def atualizar(self):
        self.x += self.velocidade * self.direcao
        if self.direcao == 1 and self.x > LARGURA_TELA + self.LARGURA:
            self.x = -self.LARGURA
        elif self.direcao == -1 and self.x < -self.LARGURA:
            self.x = LARGURA_TELA + self.LARGURA

    def obter_retangulo(self, y_camera: int) -> pygame.Rect:
        return pygame.Rect(int(self.x), self.y_tela(y_camera), self.LARGURA, self.ALTURA)

    def desenhar(self, tela: pygame.Surface, y_camera: int):
        y = self.y_tela(y_camera)
        pygame.draw.rect(tela, self.cor, (int(self.x), y, self.LARGURA, self.ALTURA), border_radius=4)
        pygame.draw.rect(tela, (180, 220, 255), (int(self.x) + 4, y + 3, 14, 9))
        pygame.draw.rect(tela, (180, 220, 255), (int(self.x) + self.LARGURA - 18, y + 3, 14, 9))


# ─────────────────────────────────────────────
#  Classe Faixa
# ─────────────────────────────────────────────
class Faixa:
    def __init__(self, y_mundo: int, tipo_faixa: str, velocidade: float = 0, direcao: int = 1):
        self.y_mundo = y_mundo
        self.tipo_faixa = tipo_faixa   # 'grass' | 'road'
        self.velocidade = velocidade
        self.direcao = direcao
        self.carros: list[Carro] = []

        if tipo_faixa == "road":
            self._popular()

    @staticmethod
    def _distancia_circular(a: float, b: float, comprimento_ciclo: float) -> float:
        distancia = abs(a - b)
        return min(distancia, comprimento_ciclo - distancia)

    def _espaco_seguro(self) -> float:
        # Espaço aumenta levemente com a velocidade para reduzir colisões visuais.
        return Carro.LARGURA + TAMANHO_CELULA * (1.0 + min(self.velocidade, TETO_VELOCIDADE_MAXIMA) * 0.15)

    def _pode_posicionar_carro(self, x: float) -> bool:
        ciclo = LARGURA_TELA + Carro.LARGURA * 2
        espaco = self._espaco_seguro()
        return all(self._distancia_circular(x, carro.x, ciclo) >= espaco for carro in self.carros)

    def _popular(self):
        quantidade = random.randint(2, 4)
        for _ in range(quantidade):
            for _ in range(200):
                x = random.uniform(-Carro.LARGURA, LARGURA_TELA + Carro.LARGURA)
                if self._pode_posicionar_carro(x):
                    self.carros.append(Carro(self.y_mundo, self.velocidade, self.direcao, x))
                    break

    def atualizar(self):
        for carro in self.carros:
            carro.atualizar()

        # Geração adaptativa para evitar congestionamento em faixas já lotadas.
        if self.tipo_faixa == "road" and len(self.carros) < 6:
            chance_geracao = max(0.0008, 0.0022 - self.velocidade * 0.00025 - len(self.carros) * 0.0002)
            if random.random() < chance_geracao:
                carro = Carro(self.y_mundo, self.velocidade, self.direcao)
                if self._pode_posicionar_carro(carro.x):
                    self.carros.append(carro)

    def desenhar(self, tela: pygame.Surface, y_camera: int):
        y = self.y_mundo - y_camera
        if y > ALTURA_TELA + TAMANHO_CELULA or y < -TAMANHO_CELULA:
            return

        if self.tipo_faixa == "grass":
            cor = GRAMA_CLARA if (self.y_mundo // TAMANHO_CELULA) % 2 == 0 else GRAMA_ESCURA
            pygame.draw.rect(tela, cor, (0, y, LARGURA_TELA, TAMANHO_CELULA))
        else:
            cor = ASFALTO_CLARO if (self.y_mundo // TAMANHO_CELULA) % 2 == 0 else ASFALTO_ESCURO
            pygame.draw.rect(tela, cor, (0, y, LARGURA_TELA, TAMANHO_CELULA))
            for x in range(0, LARGURA_TELA, TAMANHO_CELULA * 2):
                pygame.draw.rect(tela, AMARELO, (x, y + TAMANHO_CELULA // 2 - 2, TAMANHO_CELULA, 4))

        for carro in self.carros:
            carro.desenhar(tela, y_camera)


# ─────────────────────────────────────────────
#  Geração do Mundo
# ─────────────────────────────────────────────
def construir_mundo() -> list[Faixa]:
    faixas: list[Faixa] = []
    y_mundo = 0

    # meta (topo) – grama
    for _ in range(2):
        faixas.append(Faixa(y_mundo, "grass"))
        y_mundo += TAMANHO_CELULA

    secao = 0
    while len(faixas) < FAIXAS_MUNDO - 2:
        # bloco de asfalto
        quantidade_asfalto = random.randint(2, BLOCO_ASFALTO)
        for indice in range(quantidade_asfalto):
            velocidade_maxima_faixa = min(VELOCIDADE_MAXIMA_BASE + secao * 0.08, TETO_VELOCIDADE_MAXIMA)
            velocidade = random.uniform(VELOCIDADE_MINIMA_BASE, velocidade_maxima_faixa)
            direcao = 1 if indice % 2 == 0 else -1
            faixas.append(Faixa(y_mundo, "road", velocidade, direcao))
            y_mundo += TAMANHO_CELULA

        # bloco de grama
        quantidade_grama = random.randint(1, BLOCO_GRAMA + 1)
        for _ in range(quantidade_grama):
            faixas.append(Faixa(y_mundo, "grass"))
            y_mundo += TAMANHO_CELULA
        secao += 1

    # início (base) – grama
    for _ in range(2):
        faixas.append(Faixa(y_mundo, "grass"))
        y_mundo += TAMANHO_CELULA

    return faixas


# ─────────────────────────────────────────────
#  Classe Jogador / Agente
# ─────────────────────────────────────────────
class Jogador:
    TAMANHO = TAMANHO_CELULA - 10

    def __init__(self, x_mundo: int, y_mundo: int):
        self.wx = float(x_mundo)   # coordenada X (mesma na tela)
        self.wy = float(y_mundo)   # coordenada Y no mundo
        self.vivo = True
        self.pontuacao = 0         # faixas avançadas
        self.melhor_y = y_mundo
        self.cooldown_movimento = 0
        self.tiques_inativo = 0

    def y_tela(self, y_camera: int) -> int:
        return int(self.wy) - y_camera

    def obter_retangulo(self, y_camera: int) -> pygame.Rect:
        return pygame.Rect(int(self.wx) + 3, self.y_tela(y_camera) + 3, self.TAMANHO, self.TAMANHO)

    def mover(self, direcao: str):
        if self.cooldown_movimento > 0:
            return
        if direcao == "up":
            self.wy -= TAMANHO_CELULA
        elif direcao == "down":
            self.wy += TAMANHO_CELULA
        elif direcao == "left":
            self.wx -= TAMANHO_CELULA
        elif direcao == "right":
            self.wx += TAMANHO_CELULA
        self.wx = max(0, min(LARGURA_TELA - TAMANHO_CELULA, self.wx))
        self.cooldown_movimento = 12

    def atualizar(self):
        if self.cooldown_movimento > 0:
            self.cooldown_movimento -= 1

    def desenhar(self, tela: pygame.Surface, y_camera: int):
        if not self.vivo:
            return
        y = self.y_tela(y_camera)
        x = int(self.wx)
        pygame.draw.rect(tela, VERDE_SAPO, (x + 3, y + 3, self.TAMANHO, self.TAMANHO), border_radius=6)
        pygame.draw.circle(tela, BRANCO, (x + 12, y + 12), 5)
        pygame.draw.circle(tela, BRANCO, (x + 28, y + 12), 5)
        pygame.draw.circle(tela, PRETO, (x + 13, y + 12), 2)
        pygame.draw.circle(tela, PRETO, (x + 29, y + 12), 2)


# ─────────────────────────────────────────────
#  Lógica de Entradas para a Rede Neural
# ─────────────────────────────────────────────
def obter_entradas(jogador: Jogador, faixas: list[Faixa], quantidade_faixas: int) -> list[float]:
    """
    14 entradas:
      [0]  x normalizado do jogador
      [1]  y normalizado do jogador (0 = baixo, 1 = topo)
      Para cada uma das 3 faixas (atual, +1, +2 acima):
        [2+4k] dist. carro à esquerda  (normalizada)
        [3+4k] dist. carro à direita   (normalizada)
        [4+4k] velocidade da faixa     (normalizada)
        [5+4k] direção da faixa        (0 = esquerda, 1 = direita)
    """
    indice_faixa = int(jogador.wy) // TAMANHO_CELULA
    centro_jogador = jogador.wx + TAMANHO_CELULA // 2

    entradas: list[float] = [
        jogador.wx / LARGURA_TELA,
        1.0 - jogador.wy / (quantidade_faixas * TAMANHO_CELULA),
    ]

    for deslocamento in range(3):
        indice = indice_faixa - deslocamento
        if 0 <= indice < len(faixas):
            faixa = faixas[indice]
            if faixa.tipo_faixa == "road":
                distancia_esquerda = LARGURA_TELA
                distancia_direita = LARGURA_TELA
                for carro in faixa.carros:
                    centro_carro = carro.x + Carro.LARGURA // 2
                    diferenca = centro_carro - centro_jogador
                    if diferenca < 0:
                        distancia_esquerda = min(distancia_esquerda, -diferenca)
                    else:
                        distancia_direita = min(distancia_direita, diferenca)
                entradas += [
                    distancia_esquerda / LARGURA_TELA,
                    distancia_direita / LARGURA_TELA,
                    faixa.velocidade / 10.0,
                    1.0 if faixa.direcao == 1 else 0.0,
                ]
            else:
                entradas += [1.0, 1.0, 0.0, 0.0]
        else:
            entradas += [1.0, 1.0, 0.0, 0.0]

    return entradas  # sempre 14 valores


# ─────────────────────────────────────────────
#  Jogo Principal
# ─────────────────────────────────────────────
class JogoCrossyRoad:
    def __init__(self):
        pygame.init()
        self.tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
        pygame.display.set_caption("Crossy Road")
        self.fonte = pygame.font.Font(None, 36)
        self.fonte_pequena = pygame.font.Font(None, 26)
        self.cronometro = pygame.time.Clock()
        self.geracao = 0
        self.melhor_pontuacao_global = 0

    def _posicao_inicial(self, faixas: list[Faixa]):
        altura_mundo = len(faixas) * TAMANHO_CELULA
        y_mundo = altura_mundo - TAMANHO_CELULA * 3
        x_mundo = LARGURA_TELA // 2 - TAMANHO_CELULA // 2
        return x_mundo, y_mundo

    def _camera_alvo(self, melhor_y: float, altura_mundo: int) -> int:
        y_camera = int(melhor_y) - ALTURA_TELA // 2
        y_camera = max(0, min(y_camera, altura_mundo - ALTURA_TELA))
        return y_camera

    def _verificar_colisao(self, jogador: Jogador, faixas: list[Faixa], y_camera: int):
        if not jogador.vivo:
            return
        indice_faixa = int(jogador.wy) // TAMANHO_CELULA
        if 0 <= indice_faixa < len(faixas) and faixas[indice_faixa].tipo_faixa == "road":
            retangulo_jogador = jogador.obter_retangulo(y_camera)
            for carro in faixas[indice_faixa].carros:
                if retangulo_jogador.colliderect(carro.obter_retangulo(y_camera)):
                    jogador.vivo = False
                    return
        if jogador.wy > len(faixas) * TAMANHO_CELULA:
            jogador.vivo = False

    def _desenhar(self, jogadores: list[Jogador], faixas: list[Faixa], y_camera: int, vivos: int, melhor_pontuacao: int):
        self.tela.fill(GRAMA_CLARA)
        for faixa in faixas:
            faixa.desenhar(self.tela, y_camera)
        for jogador in jogadores:
            jogador.desenhar(self.tela, y_camera)

        # HUD
        pygame.draw.rect(self.tela, (20, 20, 20, 200), (8, 8, 230, 144))
        pygame.draw.rect(self.tela, BRANCO, (8, 8, 230, 144), 1)
        textos = [
            self.fonte.render(f"Geração:     {self.geracao}", True, BRANCO),
            self.fonte.render(f"Vivos:       {vivos}", True, BRANCO),
            self.fonte.render(f"Melhor score:{melhor_pontuacao}", True, BRANCO),
            self.fonte.render(f"Recorde:     {self.melhor_pontuacao_global}", True, BRANCO),
        ]
        for i, texto in enumerate(textos):
            self.tela.blit(texto, (14, 14 + i * 34))
        pygame.display.flip()

    def jogar_humano(self):
        faixas = construir_mundo()
        x_mundo, y_mundo = self._posicao_inicial(faixas)
        jogador = Jogador(x_mundo, y_mundo)
        y_camera = self._camera_alvo(y_mundo, len(faixas) * TAMANHO_CELULA)
        fonte_mensagem = pygame.font.Font(None, 48)

        executando = True
        while executando:
            self.cronometro.tick(QUADROS_POR_SEGUNDO)
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    executando = False
                if evento.type == pygame.KEYDOWN:
                    if evento.key in (pygame.K_UP, pygame.K_w):
                        jogador.mover("up")
                    if evento.key in (pygame.K_DOWN, pygame.K_s):
                        jogador.mover("down")
                    if evento.key in (pygame.K_LEFT, pygame.K_a):
                        jogador.mover("left")
                    if evento.key in (pygame.K_RIGHT, pygame.K_d):
                        jogador.mover("right")
                    if evento.key == pygame.K_r:
                        faixas = construir_mundo()
                        x_mundo, y_mundo = self._posicao_inicial(faixas)
                        jogador = Jogador(x_mundo, y_mundo)
                    if evento.key == pygame.K_ESCAPE:
                        executando = False

            if jogador.vivo:
                jogador.atualizar()
                for faixa in faixas:
                    faixa.atualizar()
                y_camera = self._camera_alvo(jogador.wy, len(faixas) * TAMANHO_CELULA)
                self._verificar_colisao(jogador, faixas, y_camera)
                if jogador.wy < jogador.melhor_y:
                    jogador.pontuacao += (jogador.melhor_y - jogador.wy) // TAMANHO_CELULA
                    jogador.melhor_y = jogador.wy

            self._desenhar([jogador], faixas, y_camera, int(jogador.vivo), jogador.pontuacao)

            if not jogador.vivo:
                mensagem = fonte_mensagem.render("Morreu! R = reiniciar", True, (255, 80, 80))
                self.tela.blit(mensagem, (LARGURA_TELA // 2 - mensagem.get_width() // 2, ALTURA_TELA // 2))
                pygame.display.flip()

        pygame.quit()

    def executar_neat(self, genomas, configuracao):
        self.geracao += 1
        faixas = construir_mundo()
        altura_mundo = len(faixas) * TAMANHO_CELULA
        x_inicial, y_inicial = self._posicao_inicial(faixas)
        y_camera = self._camera_alvo(y_inicial, altura_mundo)

        jogadores: list[Jogador] = []
        redes: list[neat.nn.FeedForwardNetwork] = []
        individuos: list[neat.DefaultGenome] = []

        for _, genoma in genomas:
            genoma.fitness = 0.0
            redes.append(neat.nn.FeedForwardNetwork.create(genoma, configuracao))
            jogadores.append(Jogador(x_inicial, y_inicial))
            individuos.append(genoma)

        tiques = 0
        PULO_QUADRO = 2

        while tiques < MAX_TIQUES and any(jogador.vivo for jogador in jogadores):
            self.cronometro.tick(QUADROS_POR_SEGUNDO)
            tiques += 1

            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if evento.type == pygame.KEYDOWN and evento.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

            for faixa in faixas:
                faixa.atualizar()

            vivos = 0
            melhor_pontuacao = 0

            for i, jogador in enumerate(jogadores):
                if not jogador.vivo:
                    continue
                vivos += 1

                jogador.atualizar()

                entradas = obter_entradas(jogador, faixas, len(faixas))
                saida = redes[i].activate(entradas)
                acao = saida.index(max(saida))

                if acao == 1:
                    jogador.mover("up")
                elif acao == 2:
                    jogador.mover("down")
                elif acao == 3:
                    jogador.mover("left")
                elif acao == 4:
                    jogador.mover("right")

                self._verificar_colisao(jogador, faixas, y_camera)

                if jogador.wy > y_inicial:
                    individuos[i].fitness -= 3.0
                    jogador.vivo = False
                    continue

                avanco = (y_inicial - jogador.wy) / TAMANHO_CELULA
                jogador.pontuacao = int(avanco)
                melhor_pontuacao = max(melhor_pontuacao, jogador.pontuacao)
                self.melhor_pontuacao_global = max(self.melhor_pontuacao_global, melhor_pontuacao)

                # Fitness prioriza score (avanço). Sobrevivência conta pouco.
                individuos[i].fitness = avanco * 3.0 + tiques * 0.0002

                if jogador.wy >= jogador.melhor_y:
                    jogador.tiques_inativo += 1
                    if jogador.tiques_inativo > 200:
                        individuos[i].fitness -= 3.0
                        jogador.vivo = False
                        continue
                else:
                    jogador.tiques_inativo = 0
                    jogador.melhor_y = jogador.wy

            jogadores_vivos = [jogador for jogador in jogadores if jogador.vivo]
            if jogadores_vivos:
                melhor_y = min(jogador.wy for jogador in jogadores_vivos)
                y_camera = self._camera_alvo(melhor_y, altura_mundo)

            if tiques % PULO_QUADRO == 0:
                self._desenhar(jogadores, faixas, y_camera, vivos, melhor_pontuacao)

        for i, jogador in enumerate(jogadores):
            avanco = (y_inicial - jogador.wy) / TAMANHO_CELULA
            individuos[i].fitness = max(individuos[i].fitness, avanco * 3.0)


# ─────────────────────────────────────────────
#  Ponto de Entrada
# ─────────────────────────────────────────────
def executar_ia():
    logger = logging.getLogger(__name__)
    caminho_configuracao = os.path.join(os.path.dirname(__file__), "neat_config.txt")
    if not os.path.exists(caminho_configuracao):
        logger.error(f"Arquivo de configuração não encontrado: {caminho_configuracao}")
        sys.exit(1)

    configuracao = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        caminho_configuracao,
    )
    populacao = neat.Population(configuracao)

    # Reinicia a busca a partir do melhor genoma salvo (se existir).
    melhor_salvo = carregar_melhor_genoma()
    if melhor_salvo is not None:
        try:
            populacao.population[0] = melhor_salvo
            populacao.species.speciate(configuracao, populacao.population, populacao.generation)
            logger.info("Melhor genoma anterior carregado como base da nova execucao.")
        except Exception as erro:
            logger.warning(f"Nao foi possivel usar o genoma salvo: {erro}")

    # Captura stdout para logar
    stdout_antigo = sys.stdout
    buffer_saida = StringIO()
    sys.stdout = buffer_saida

    try:
        populacao.add_reporter(neat.StdOutReporter(True))
        populacao.add_reporter(neat.StatisticsReporter())

        jogo = JogoCrossyRoad()

        # Executa indefinidamente, uma geracao por vez.
        while True:
            vencedor = populacao.run(jogo.executar_neat, 1)

            # Recupera saida do NEAT e limpa o buffer para evitar crescimento infinito.
            saida = buffer_saida.getvalue()
            buffer_saida.seek(0)
            buffer_saida.truncate(0)

            for linha in saida.split("\n"):
                if linha.strip():
                    logger.info(linha)

            logger.info("=== Melhor genoma encontrado ===")
            logger.info(str(vencedor))
            salvar_melhor_genoma(vencedor)
    except KeyboardInterrupt:
        sys.stdout = stdout_antigo
        logger.info("Treinamento interrompido pelo usuario.")
    except Exception as erro:
        sys.stdout = stdout_antigo
        logger.error(f"Erro: {erro}")
        raise


def jogar_humano():
    jogo = JogoCrossyRoad()
    jogo.jogar_humano()


if __name__ == "__main__":
    configurar_log()

    print("Crossy Road")
    print("  1) Jogar manualmente  (setas / WASD | R = reiniciar)")
    print("  2) Treinar IA (NEAT)")
    escolha = input("Escolha [1/2]: ").strip()
    if escolha == "1":
        jogar_humano()
    else:
        executar_ia()
