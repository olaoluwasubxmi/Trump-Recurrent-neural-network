{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import re\n",
    "import random\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense\n",
    "from keras.layers import LSTM\n",
    "from keras.layers import Dropout\n",
    "from keras.models import Model \n",
    "from keras.layers import Input, Activation, Embedding, LSTM\n",
    "from keras.optimizers import Adam\n",
    "from keras.optimizers import RMSprop\n",
    "from keras.callbacks import LambdaCallback\n",
    "from keras.preprocessing.text import Tokenizer\n",
    "from keras.utils import to_categorical\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"donaldtweets.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Convert to Lower case\n",
    "text = df['Tweet_Text'].str.lower()\n",
    "\n",
    "## Remove the URLs\n",
    "text = text.map(lambda s: ' '.join([x for x in s.split() if 'http' not in x]))\n",
    "\n",
    "## Remove short tweets\n",
    "text = text[text.map(len)>40]\n",
    "\n",
    "## Remove emojis\n",
    "text = text.apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total Vocabulary:  10760\n"
     ]
    }
   ],
   "source": [
    " tokenizer = Tokenizer()\n",
    " tokenizer.fit_on_texts(text)\n",
    " vocab_size = len(tokenizer.word_counts) + 1\n",
    " print(\"Total Vocabulary: \", vocab_size )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total Row Count:  6886\n",
      "Training Data Count:  5508\n",
      "Test Data Coount:  1378\n"
     ]
    }
   ],
   "source": [
    " N = text.shape[0]\n",
    " print(\"Total Row Count: \", N)\n",
    " prop_train = 0.8\n",
    " train = int(N*prop_train)\n",
    " print(\"Training Data Count: \", train)\n",
    " test = N - train\n",
    " print(\"Test Data Coount: \", test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total Sequences: 114825\n"
     ]
    }
   ],
   "source": [
    "sequences, index_train, index_test = [], [], []\n",
    "count = 0\n",
    "for irow,line in enumerate(text):\n",
    "    #print(irow, line)\n",
    "    encoded = tokenizer.texts_to_sequences([line])[0]    \n",
    "    #print(encoded)\n",
    "    for i in range(1, len(encoded)):\n",
    "        sequence = encoded[:i+1]\n",
    "        sequences.append(sequence)\n",
    "        if irow < train:        \n",
    "            index_train.append(count)     \n",
    "        else:         \n",
    "            index_test.append(count)\n",
    "        count += 1\n",
    "print('Total Sequences: %d' % (len(sequences)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Max Sequence Length: 32\n"
     ]
    }
   ],
   "source": [
    "from keras_preprocessing.sequence import pad_sequences\n",
    "max_length = max([len(seq) for seq in sequences])\n",
    "sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')\n",
    "print('Max Sequence Length: %d' % (max_length))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(114825,)\n",
      "(92288, 31)\n",
      "(92288, 10760)\n"
     ]
    }
   ],
   "source": [
    "sequences = np.array(sequences)\n",
    "X, y = sequences[:,:-1],sequences[:,-1]\n",
    "print(y.shape)\n",
    "y = to_categorical(y, num_classes=vocab_size)\n",
    "X_train, y_train, X_test, y_test = X[index_train], y[index_train],X[index_test],  y[index_test]\n",
    "print(X_train.shape)\n",
    "print(y_train.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_model(vocab_size,\n",
    "                  input_length=1,\n",
    "                  dim_dense_embedding=10,\n",
    "                  hidden_unit_LSTM=5):\n",
    "    main_input = Input(shape=(input_length,),dtype='int32',name='main_input')\n",
    "    embedding = Embedding(vocab_size, dim_dense_embedding, \n",
    "                          input_length=input_length)(main_input)\n",
    "    x = LSTM(hidden_unit_LSTM)(embedding)\n",
    "    main_output = Dense(vocab_size, activation='softmax')(x)\n",
    "    model = Model(inputs=[main_input],\n",
    "                   outputs=[main_output])\n",
    "    return(model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "ename": "IndentationError",
     "evalue": "unexpected indent (Temp/ipykernel_16052/3173069803.py, line 6)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"C:\\Users\\Subxmi\\AppData\\Local\\Temp/ipykernel_16052/3173069803.py\"\u001b[1;36m, line \u001b[1;32m6\u001b[0m\n\u001b[1;33m    model.compile(loss='categorical_crossentropy',\u001b[0m\n\u001b[1;37m    ^\u001b[0m\n\u001b[1;31mIndentationError\u001b[0m\u001b[1;31m:\u001b[0m unexpected indent\n"
     ]
    }
   ],
   "source": [
    "model = build_model(vocab_size,\n",
    " input_length=X.shape[1],\n",
    " dim_dense_embedding=30\n",
    " hidden_unit_LSTM=64)\n",
    "\n",
    " ##Compile network\n",
    "    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
    "    ##fit network\n",
    "    tf_model = model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=20, verbose=2, batch_size=128)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  },
  "vscode": {
   "interpreter": {
    "hash": "285ddf65587f3c042e287c3856e7d92f4bb8cbaf58923bfd87e25b328822367e"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
