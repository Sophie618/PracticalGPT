import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ===================== 1. 数据预处理（简单英中翻译数据集）=====================
# 语料库：英文→中文
corpus = [
    ("I love you", "我 爱 你"),
    ("I like NLP", "我 喜欢 NLP"),
    ("Seq2seq is good", "序列到序列 很 好"),
    ("Attention is useful", "注意力 机制 很 有用")
]

# 构建词表（源语言：英文；目标语言：中文）
def build_vocab(corpus, is_source):
    vocab = {"<pad>":0, "<start>":1, "<end>":2}  # 特殊标记
    idx = 3
    for pair in corpus:
        sentence = pair[0] if is_source else pair[1]
        for token in sentence.split():
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab, {v:k for k,v in vocab.items()}

# 源语言（英文）词表和逆词表
src_vocab, src_idx2word = build_vocab(corpus, is_source=True)
# 目标语言（中文）词表和逆词表
tgt_vocab, tgt_idx2word = build_vocab(corpus, is_source=False)

# 序列转数字ID（加<start>/<end>，填充）
def seq2id(sentence, vocab, max_len, is_source):
    tokens = sentence.split()
    if is_source:
        ids = [vocab[t] for t in tokens if t in vocab]
    else:
        ids = [vocab["<start>"]] + [vocab[t] for t in tokens if t in vocab] + [vocab["<end>"]]
    # 填充到max_len
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)

# 超参数
MAX_SRC_LEN = 5  # 源序列最大长度
MAX_TGT_LEN = 6  # 目标序列最大长度
EMBEDDING_DIM = 16  # 嵌入维度
HIDDEN_DIM = 32  # LSTM隐藏层维度
BATCH_SIZE = 2
EPOCHS = 100

# 构建数据集
src_data = [seq2id(pair[0], src_vocab, MAX_SRC_LEN, is_source=True) for pair in corpus]
tgt_data = [seq2id(pair[1], tgt_vocab, MAX_TGT_LEN, is_source=False) for pair in corpus]
dataset = torch.utils.data.TensorDataset(torch.stack(src_data), torch.stack(tgt_data))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===================== 2. 定义注意力层（Bahdanau Attention）=====================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_hiddens, decoder_hidden):
        # encoder_hiddens: [seq_len, batch_size, hidden_dim]
        # decoder_hidden: [1, batch_size, hidden_dim]
        score = self.Va(torch.tanh(self.Wa(encoder_hiddens) + self.Ua(decoder_hidden)))  # [seq_len, batch_size, 1]
        attn_weights = torch.softmax(score, dim=0)  # 注意力权重 [seq_len, batch_size, 1]
        context = (encoder_hiddens * attn_weights).sum(dim=0)  # 上下文向量 [batch_size, hidden_dim]
        return context, attn_weights

# ===================== 3. 定义编码器（Encoder-LSTM）=====================
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)

    def forward(self, src):
        # src: [batch_size, seq_len] → 转置为 [seq_len, batch_size]
        src = src.permute(1, 0)
        embed = self.embedding(src)  # [seq_len, batch_size, embedding_dim]
        encoder_hiddens, (hidden, cell) = self.lstm(embed)  # encoder_hiddens: [seq_len, batch_size, hidden_dim]
        return encoder_hiddens, (hidden, cell)

# ===================== 4. 定义解码器（Decoder-LSTM+Attention）=====================
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=False)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, tgt_vocab_size)

    def forward(self, tgt, encoder_hiddens, hidden, cell):
        # tgt: [batch_size, 1] → 转置为 [1, batch_size]
        tgt = tgt.permute(1, 0)
        embed = self.embedding(tgt)  # [1, batch_size, embedding_dim]
        # 注意力机制获取上下文向量
        context, attn_weights = self.attention(encoder_hiddens, hidden)  # context: [batch_size, hidden_dim]
        # 拼接嵌入向量和上下文向量
        lstm_input = torch.cat([embed, context.unsqueeze(0)], dim=2)  # [1, batch_size, embedding_dim+hidden_dim]
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))  # [1, batch_size, hidden_dim]
        # 拼接lstm输出和上下文向量，做最终预测
        output = self.fc(torch.cat([lstm_out.squeeze(0), context], dim=1))  # [batch_size, tgt_vocab_size]
        return output, (hidden, cell), attn_weights

# ===================== 5. 定义Seq2seq整体模型=====================
class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        batch_size = src.shape[0]
        tgt_vocab_size = self.decoder.fc.out_features
        tgt_len = tgt.shape[1]
        # 存储解码器所有输出
        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(src.device)
        # 编码器前向传播
        encoder_hiddens, (hidden, cell) = self.encoder(src)
        # 解码器初始输入：tgt的第一个token（<start>）
        dec_input = tgt[:, 0:1]
        # 逐步解码
        for t in range(1, tgt_len):
            output, (hidden, cell), _ = self.decoder(dec_input, encoder_hiddens, hidden, cell)
            outputs[t] = output
            # 取预测概率最大的token作为下一个输入（教师强制：也可直接用tgt[:,t:t+1]加速训练）
            dec_input = output.argmax(1).unsqueeze(1)
        return outputs

# ===================== 6. 模型初始化+优化器+损失函数=====================
# 设备：GPU优先，无则CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化编码器和解码器
encoder = Encoder(len(src_vocab), EMBEDDING_DIM, HIDDEN_DIM).to(device)
decoder = Decoder(len(tgt_vocab), EMBEDDING_DIM, HIDDEN_DIM).to(device)
# 初始化Seq2seq模型
model = Seq2seq(encoder, decoder).to(device)
# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 损失函数：忽略填充位（padding_idx=0）
criterion = nn.CrossEntropyLoss(ignore_index=0)

# ===================== 7. 模型训练=====================
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        # 前向传播
        outputs = model(src, tgt)  # [tgt_len, batch_size, tgt_vocab_size]
        # 调整维度计算损失：outputs→[tgt_len-1, batch_size, vocab_size]；tgt→[batch_size, tgt_len-1]
        loss = criterion(
            outputs[1:].reshape(-1, len(tgt_vocab)),
            tgt[:, 1:].reshape(-1)
        )
        # 反向传播+优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {total_loss/len(dataloader):.4f}")

# ===================== 8. 模型推理（预测新句子）=====================
def predict(model, src_sentence, src_vocab, tgt_vocab, tgt_idx2word, max_len, device):
    model.eval()
    with torch.no_grad():
        # 预处理输入句子
        src = seq2id(src_sentence, src_vocab, max_len, is_source=True).unsqueeze(0).to(device)
        # 编码器前向传播
        encoder_hiddens, (hidden, cell) = model.encoder(src)
        # 解码器初始输入：<start>
        dec_input = torch.tensor([[tgt_vocab["<start>"]]]).to(device)
        # 存储预测结果
        pred_ids = []
        for _ in range(max_len):
            output, (hidden, cell), _ = model.decoder(dec_input, encoder_hiddens, hidden, cell)
            pred_id = output.argmax(1).item()
            pred_ids.append(pred_id)
            # 生成<end>则停止
            if pred_id == tgt_vocab["<end>"]:
                break
            # 下一个输入为当前预测的token
            dec_input = torch.tensor([[pred_id]]).to(device)
        # 转换为中文token
        pred_tokens = [tgt_idx2word[id] for id in pred_ids if id not in [0,1,2]]
    return " ".join(pred_tokens)

# 测试预测
test_sentences = ["I love you", "I like NLP", "Seq2seq is good"]
for sent in test_sentences:
    pred = predict(model, sent, src_vocab, tgt_vocab, tgt_idx2word, MAX_TGT_LEN, device)
    print(f"输入：{sent} → 预测：{pred}")