import streamlit as st
import pandas as pd
import joblib
import cloudpickle
import re
from sklearn.preprocessing import FunctionTransformer

# ==================================================
# 0. DEFINIÇÕES OBRIGATÓRIAS (MODELO E DADOS)
# ==================================================
serie_colunas_ordinais = {
    'Frequencia_de_consumo_de_vegetais_nas_refeicoes': {
        'Raramente': 0, 'As_vezes': 1, 'Sempre': 2
    },
    'Numero_de_refeicoes_principais_por_dia': {
        'Uma_refeicao': 0, 'Duas': 1, 'Tres': 2, 'Quatro_refeicao_ou_mais': 3
    },
    'Consumo_de_lanches_entre_as_refeicoes': {
        'Nao_consome': 0, 'As_vezes': 1, 'Frequentemente': 2, 'Sempre': 3
    },
    'Consumo_diario_de_agua': {
        '<_1_L/dia': 0, '1–2_L/dia': 1, '>_2_L/dia': 2
    },
    'Frequencia_semanal_de_atividade_fisica': {
        'Nenhuma': 0, '~1–2×/sem': 1, '~3–4×/sem': 2, '5×/sem_ou_mais': 3
    },
    'Tempo_diario_usando_dispositivos_eletronicos': {
        '~0–2_h/dia': 0, '~3–5_h/dia': 1, '>_5_h/dia': 2
    },
    'Consumo_de_bebida_alcoolica': {
        'Nao_bebe': 0, 'As_vezes': 1, 'Frequentemente': 2, 'Sempre': 3
    }
}

def aplicar_serie_ordinais(dataframe):
    df_ = dataframe.copy()
    for col, mapa in serie_colunas_ordinais.items():
        if col in df_.columns:
            df_[col] = df_[col].map(mapa).fillna(0)
    return df_

# ==================================================
# 1. CONFIGURAÇÃO VISUAL
# ==================================================
st.set_page_config(page_title="Triagem de Obesidade", page_icon="🏥", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f5f7f9;}
    /* Remove setas dos campos numéricos */
    input[type=number]::-webkit-inner-spin-button,
    input[type=number]::-webkit-outer-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    /* Botão Vermelho Grande */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        font-weight: bold;
        background-color: #ff4b4b;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# 2. FUNÇÕES ÚTEIS
# ==================================================
def limpar_telefone(tel_bruto):
    if not tel_bruto: return None
    apenas_numeros = re.sub(r'\D', '', str(tel_bruto))
    if len(apenas_numeros) < 10 or len(apenas_numeros) > 11:
        return None
    return apenas_numeros

def formatar_telefone_visual(tel_limpo):
    if not tel_limpo: return ""
    if len(tel_limpo) == 11:
        return f"({tel_limpo[:2]}) {tel_limpo[2:7]}-{tel_limpo[7:]}"
    if len(tel_limpo) == 10:
        return f"({tel_limpo[:2]}) {tel_limpo[2:6]}-{tel_limpo[6:]}"
    return tel_limpo

# ==================================================
# 3. CARGA DE ARTEFATOS
# ==================================================
@st.cache_resource
def carregar_dados():
    try:
        with open('pipeline_modelo_obesidade.pkl', 'rb') as f:
            pipeline = cloudpickle.load(f)
        metadados = joblib.load('preset_metadados_obesidade.pkl')
        return pipeline, metadados
    except:
        return None, None

pipeline, metadados = carregar_dados()

if pipeline is None:
    st.error("🚨 ERRO CRÍTICO: Modelos não encontrados. Rode o treinamento no Colab primeiro.")
    st.stop()

opcoes = serie_colunas_ordinais

# ==================================================
# 4. INTERFACE DO USUÁRIO
# ==================================================
st.title("🏥 Sistema de Triagem (IA)")
st.markdown("**Atenção:** Todos os campos são de preenchimento obrigatório.")
st.divider()

# Nota: Não usamos st.form para impedir o envio com "Enter".
# Usamos index=None para Selectbox e value=None para NumberInput para forçar vazio.

# --- BLOCO 1: DADOS PESSOAIS ---
st.subheader("👤 Dados Pessoais")
c1, c2, c3 = st.columns([1.5, 1.5, 1.5])

nome = c1.text_input("Nome", placeholder="Digite seu nome")
sobrenome = c2.text_input("Sobrenome", placeholder="Digite seu sobrenome")
celular_input = c3.text_input("Celular (DDD + Número)", placeholder="Ex: 11999999999")

# --- BLOCO 2: BIOMETRIA ---
st.subheader("📏 Biometria")
b1, b2, b3, b4 = st.columns(4)

sexo = b1.selectbox("Sexo Biológico", ["Homem", "Mulher"], index=None, placeholder="Selecione...")
idade = b2.number_input("Idade", min_value=14, max_value=100, step=1, value=None, placeholder="Anos")
altura = b3.number_input("Altura (m)", min_value=1.40, max_value=2.50, step=0.01, value=None, placeholder="Ex: 1.70")
peso = b4.number_input("Peso (kg)", min_value=30.0, max_value=300.0, step=0.1, value=None, placeholder="Ex: 80.0")

# --- BLOCO 3: HÁBITOS (Selectboxes VAZIOS por padrão) ---
st.subheader("🍎 Hábitos de Vida")
h1, h2, h3 = st.columns(3)

with h1:
    hist = st.selectbox("Histórico Familiar Obesidade", ["Ha_historico", "Nao_ha"], index=None, placeholder="Selecione...")
    caloricos = st.selectbox("Consome Hipercalóricos?", ["Sim", "Nao"], index=None, placeholder="Selecione...")
    fumar = st.selectbox("Hábito de Fumar", ["Fuma", "Nao_fuma"], index=None, placeholder="Selecione...")
    monitora = st.selectbox("Monitora Calorias?", ["Sim", "Nao"], index=None, placeholder="Selecione...")

with h2:
    agua = st.selectbox("Consumo de Água", list(opcoes['Consumo_diario_de_agua'].keys()), index=None, placeholder="Selecione...")
    alcool = st.selectbox("Consumo de Álcool", list(opcoes['Consumo_de_bebida_alcoolica'].keys()), index=None, placeholder="Selecione...")
    fisica = st.selectbox("Atividade Física", list(opcoes['Frequencia_semanal_de_atividade_fisica'].keys()), index=None, placeholder="Selecione...")
    telas = st.selectbox("Tempo em Telas", list(opcoes['Tempo_diario_usando_dispositivos_eletronicos'].keys()), index=None, placeholder="Selecione...")

with h3:
    vegetais = st.selectbox("Vegetais nas Refeições", list(opcoes['Frequencia_de_consumo_de_vegetais_nas_refeicoes'].keys()), index=None, placeholder="Selecione...")
    refeicoes = st.selectbox("Refeições Principais/Dia", list(opcoes['Numero_de_refeicoes_principais_por_dia'].keys()), index=None, placeholder="Selecione...")
    lanches = st.selectbox("Lanches entre Refeições", list(opcoes['Consumo_de_lanches_entre_as_refeicoes'].keys()), index=None, placeholder="Selecione...")
    transporte = st.selectbox("Meio de Transporte", ["A_pe", "Bicicleta", "Carro", "Moto", "Transporte_publico"], index=None, placeholder="Selecione...")

st.markdown("---")
# Botão fora de formulário para evitar acionamento por Enter
submit = st.button("CADASTRAR E CALCULAR")

# ==================================================
# 5. VALIDAÇÃO E LÓGICA
# ==================================================
if submit:
    # Lista para acumular erros
    erros = []

    # 1. Validação de Campos Vazios (None ou String Vazia)
    # Verifica Textos
    if not nome or not nome.strip(): erros.append("Campo 'Nome' é obrigatório.")
    if not sobrenome or not sobrenome.strip(): erros.append("Campo 'Sobrenome' é obrigatório.")

    # Verifica Numéricos (None)
    if idade is None: erros.append("Campo 'Idade' é obrigatório.")
    if altura is None: erros.append("Campo 'Altura' é obrigatório.")
    if peso is None: erros.append("Campo 'Peso' é obrigatório.")

    # Verifica Selectboxes (None)
    if sexo is None: erros.append("Campo 'Sexo' é obrigatório.")
    if hist is None: erros.append("Campo 'Histórico Familiar' é obrigatório.")
    if caloricos is None: erros.append("Campo 'Hipercalóricos' é obrigatório.")
    if fumar is None: erros.append("Campo 'Fuma?' é obrigatório.")
    if monitora is None: erros.append("Campo 'Monitora Calorias' é obrigatório.")
    if agua is None: erros.append("Campo 'Água' é obrigatório.")
    if alcool is None: erros.append("Campo 'Álcool' é obrigatório.")
    if fisica is None: erros.append("Campo 'Atividade Física' é obrigatório.")
    if telas is None: erros.append("Campo 'Tempo Telas' é obrigatório.")
    if vegetais is None: erros.append("Campo 'Vegetais' é obrigatório.")
    if refeicoes is None: erros.append("Campo 'Refeições' é obrigatório.")
    if lanches is None: erros.append("Campo 'Lanches' é obrigatório.")
    if transporte is None: erros.append("Campo 'Transporte' é obrigatório.")

    # 2. Validação Específica do Celular
    tel_limpo = limpar_telefone(celular_input)
    if not celular_input:
        erros.append("Campo 'Celular' é obrigatório.")
    elif not tel_limpo:
        erros.append("Celular inválido. Digite DDD + Número (ex: 11999999999).")

    # --- DECISÃO: MOSTRAR ERRO OU CALCULAR ---
    if erros:
        st.error("⚠️ Formulário incompleto ou inválido. Verifique os campos abaixo:")
        for e in erros:
            st.error(f"• {e}")
    else:
        # Se chegou aqui, TODOS os campos estão preenchidos corretamente
        try:
            # Formatação Visual
            nome_completo = f"{nome.strip()} {sobrenome.strip()}"
            tel_formatado = formatar_telefone_visual(tel_limpo)

            # Monta DataFrame para a IA
            dados = pd.DataFrame([{
                'Idade_em_anos': idade, 'Sexo_biologico': sexo, 'Altura_em_metros': altura,
                'Peso_em_quilogramas': peso, 'Historico_familiar_de_excesso_de_peso': hist,
                'Consumo_frequente_de_alimentos_muito_caloricos': caloricos,
                'Frequencia_de_consumo_de_vegetais_nas_refeicoes': vegetais,
                'Numero_de_refeicoes_principais_por_dia': refeicoes,
                'Consumo_de_lanches_entre_as_refeicoes': lanches, 'Habito_de_fumar': fumar,
                'Consumo_diario_de_agua': agua, 'Monitora_a_ingestao_calorica_diaria': monitora,
                'Frequencia_semanal_de_atividade_fisica': fisica,
                'Tempo_diario_usando_dispositivos_eletronicos': telas,
                'Consumo_de_bebida_alcoolica': alcool, 'Meio_de_transporte_habitual': transporte
            }])

            # 1. Cálculo do IMC
            imc = peso / (altura ** 2)

            # 2. Predição da IA
            pred = pipeline.predict(dados)[0]
            mapa_res = {
                0: 'Abaixo do Peso', 1: 'Peso Normal', 2: 'Sobrepeso I',
                3: 'Sobrepeso II', 4: 'Obesidade I', 5: 'Obesidade II',
                6: 'Obesidade III'
            }
            res_txt = mapa_res.get(pred, "Desconhecido")

            # 3. Exibição
            st.balloons()
            st.success("✅ Cadastro e Análise realizados com sucesso!")

            st.markdown(f"**Paciente:** {nome_completo} | **Celular:** {tel_formatado}")

            # Métricas Lado a Lado
            c_res1, c_res2 = st.columns(2)
            c_res1.metric("IMC Calculado", f"{imc:.2f}")
            c_res2.metric("Classificação IA", res_txt, delta_color="inverse")

            # Mensagens de Apoio
            if "Obesidade" in res_txt:
                st.warning("⚠️ **Atenção:** O perfil comportamental e biométrico indica Obesidade. Recomendamos procurar um médico.")
            elif "Sobrepeso" in res_txt:
                st.info("ℹ️ **Nota:** Indicativo de Sobrepeso. Pequenas mudanças de hábito podem ajudar.")
            else:
                st.success("🎉 **Parabéns:** Seus indicadores estão saudáveis!")

        except Exception as e:
            st.error(f"Erro interno ao processar dados: {e}")
