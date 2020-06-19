# Regressione Lineare

## Modello di conoscenza

Funzione che associa ad ogni dato in input una classe/valore numerico.

## Fasi del learning supervisionato

1. Raccolta dei dati, analisi esplorativa e della loro qualità
2. Trattamento di valori mancanti, normalizzazione, standardizzazione
3. Selezione di feature, ossia variabili rilevanti rispetto agli obiettivi
4. Divisione dei dati in training set, validation set e test set
5. Estrazione di modelli di conoscenza dal training set con algoritmi di Machine Learning individuando i parametri che massimizzino l'accuratezza sul validation set
6. Deployment della conoscenza in applicazioni

## Predire variabili continue

La regressione lineare stima da un data set una funzione lineare che associ variabili indipendenti di input ad una variabile dipendente di output.  
La funzione ha una serie di parametri i cui valori devono essere determinati in modo:
- **diretto**: costo O(N<sup>3</sup>)
- **numerico**: si usa la discesa del gradiente, che non garantisce soluzioni ottime ma è parallelizzabile, in crementale e anche per problemi non convessi

## Regressione lineare univariata

La regressione univariata consente di stimare una variabile dipendente y sulla base di un'unica variabile indipendente x. Ci si aspetta quindi una funzione nella forma tipica della retta, y = ax+b.  
L'obiettivo dell'analisi di regressione è trovare i valori dei parametri a e b che rendano la stima più accurata possibile. Dobbiamo individuare i valori che diano la migliore approssimazione possibile.  

### Misura dell'errore

Come si può misurare l'errore complessivo sull'insieme di predizioni? Con la media dei quadrati dei singoli errori, Mean Squared Error.  
Fissato un set di dati, l'errore è calcolato come una funziona continua sui parametri del modello di predizione.  
L'obiettivo della regressione è quello di trovare i parametri per cui il valore della funzione d'errore sia minimo.

### Gradiente di una funzione

Il gradiente è il vettore delle derivate parziali della funzione f per ciascuna delle sue variabili.  
Il gradiente calcolato in un punto x indica l'inclinazione della curva nel punto stesso.

#### Discesa del gradiente

La discesa del gradiente è un metodo iterativo per trovare un minimo locale di una funzione seguendo il suo gradiente.  
La lunghezza del passo di discesa, step size, è un iperparametro dell'algoritmo di discesa da impostare. Noto nelle reti neurali come larning rate.  
Applicando quindi la discesa del gradiente sulla funzione d'errore, possiamo trovare i valori dei parametri per cui è minima.  
Con la regressione possiamo quindi usare i dati disponibili per ottenere in automatico una funzione che predica la nostra variabile continua. 

## Regressione lineare multivariata

Con la regressione multivariata si stima una variabile dipendete y sulla base di più variabili indipendenti x1,...,xn.  
Nella regressione lineare multivariata si considera la variabile obiettivo come una somma pesata delle sue variabili indipendenti:
- La regressione determina il peso di ogni variabile numerica ed ogni variabile categorica
- La variabile obiettivo è la somma delle variabili numeriche note moltiplicate per i rispettivi pesi dei parametri, compresi quelli categorici.

# Normalizzazione dei dati

In generale, le variabili coinvolte in un modello di regressione possono utilizzare scale di valori molto diverse, questo rende difficile la convergenza della discesa del gradiente.  
Possiamo eseguire la discesa del gradiente su dati normalizzati e "denormalizzare" i parametri ottenuti.  
Addestrando un modello su dati normalizzati, i suoi parametri saranno ottimizzati su di essi. Per usare il modello occorre quindi coerentemente normalizzare gli input e dernomalizzare gli output ottenuti.

## Valutazione del modello di regressione

Un modello di regressione è addestrato in modo da minimizzare l'errore su un insieme di dati noti e il suo obiettivo è ottenere un modello generale. Se l'accuratezza su nuovi dati è significativamente più bassa allora ci possono essere due problemi: il modello è troppo aderente ai dati di addestramento perdendo di generalità (overfitting) o il training set impiegato non è rappresentativo dei nuovi dati.  

### Valutazione con metodo Hold-Out

Il metodo hold-out prevede di dividere i dati a disposizione in due insiemi, secondo una proporzione predefinita:
- Training set: utilizzato per addestrare il modello di regressione, minimizzando l'errore su di esso
- Validation set: usato dopo l'addestramento per verificare l'errore del modello su dati ignoti

### Errore relativo

L'errore quadratico medio è funzionale all'addestramento del modello di regressione tramite gradiente, possiamo in aggiunta valutare l'errore secondi altri criteri, come ad esempio l'errore relativo tra un valore y reale e la stima y.

### Coefficiente di determinazione R^2

Il coefficiente di determinazione R^2 indica la proporzione tra variabilità dei dati e correttezza del modello. Il suo valore è compreso tra 0 (nessuna relazione) e 1 (massima relazione).

# K-Fold cross validation

È un metodo alternativo al metodo di hold-out per la divisione del dataset in training e validation set. Consiste nel suddividere i dati in k sottoinsieme disgiunti dove un sottoinsieme è usato com validation set e gli altri k-1 come training set. Si ripete k volte con ogni k subset.  
Esiste anche la k-fold cross validation stratificata, che è come la suddetta ma con stessa distribuzione e caratteristiche dei dati in ogni fold.

# Nested cross validation

È una versione migliorata della K-Fold cross validation. Ogni parte di training della k-fold cs esterna è suddivisa nella cs intena in m-subfold che sono usati per individuare gli iperparametri migliori.  
Gli iperparametri migliori sono poi usati per addestrare e testare il modello nella relativa parte della validation esterna.

# Regressione non lineare

## Regressione polinomiale

La regressione polinomiale è una generalizzazione di quella lineare con altri termini di grado superiore, utile per ottenere modelli capaci di descrivere data set più complessi.

### Complessità del modello

All'aumentare della complessità del modello di learning si riduce l'errore, ma dopo una certa soglia di complessità l'errore sul validation set torna a crecere: questa è la soglia di overfitting. Il modello può essere:
- overfitting: il modello è troppo complesso
- underfitting: il modello è troppo semplice
- optimal: minimizza l'errore sul validation set

Dal training set si determinano i parametri _theta_ del modello di learning mentre gli iperparametri sono tutte le altre scelte e si determinano dal validation set.

### Regolarizzazione

Il grado nella regressione polinomiale misura la complessità del modello di learning ma i coefficienti del polinomio diventano molto grandi e ciò provoca forti oscillazioni nella regressione peggiorandone l'accuratezza.  
Regolarizzare significa ridurre il valore dei coefficienti del polinomio.

## Regressione Ridge

Nella regressione ridge, si aggiunge alla funzione d'errore da minimizzare anche _lambda_*_theta_. _Lambda_ è un iperaparametro, compreso tra 0 e infinito ed è l'indice di regolarizzazione: se è 0 non si ha regolarizzazione mentre più è grande, più si riducono i coefficienti _theta_. Il metodo più semplice per determinarlo è utilizzare la k-fold cross validation.  
La regressione ridge utilizza la normale L2 per minimizzare l'errore.

## Collinearità: dipendenze tra variabili di input

Nella regressione ordinaria si assume che le variabili di input siano indipendenti tra loro poiché con dipendenze la regressione sarebbe instabile.  
Se due variabili sono in qualche modo dipendenti l'una dall'altra, si ha un effetto noto nel Statistical Learning come interazione.  
La regolarizzazione risolve il problema poiché vincola/riduce le possibili soluzione.

## Regressione LASSO

Se penalizziamo i _theta_ nella minimizzazione dell'errore con norma L1 invece che L2, la soluzione _theta*_ è vincolata all'interno di un ipercubo centrato sull'origine: maggiore è _lambda_, più che probabile che _theta*_ cada sugli spigoli azzerando diversi _theta_ e si eliminano più variabili irrilevanti, dando luogo a una soluzione sparsa. Il modello predittivo, eliminare le variabili ritenute irrilevanti, risulterà più interpretabile.

## Elastic net

La regressione elastic net generalizza ridge e LASSO con entrambe le penalizzazioni dei _theta_ con norma L1 e L2. Introduce un secondo iperparametro _alpha_ per pesare le penalizzazioni: con _alpha_ a 0 si ha la regressione ridge, con _alpha_ a 1 si ha la regressione LASSO. Valori intermedi combinano le due penalizzazioni.

## Esplosione della dimensionalità

Si hanno problemi ad elevata dimensionalità se il numero di variabili è maggiore, o dello stesso ordine di grandezza, rispetto al numero di istanze.  
Quanti parametri _theta_ si hanno con n variabili e grado g? (n+g g) termini.  
Una regola generale è quella di diminuire g all'aumentare di n e viceversa.

### Funzioni kernel

Si può mappare i dati originali in un nuovo spazio ad elevata dimensionalità senza creare le relative nuove variabili utilizzato un metodo chiamato kernel trick che vale per ogni grado g e numero n di dimensioni.  
Il metodo serve a portare i dati in nuovi spazi ad elevata dimensionalità senza creare nuove variabili.  
L'unico limite è che possono generare modelli affetti da overfitting.  

#### Gaussian Radial basis RBF

È una funzione non lineare più complessa della polinomiale. Restituisce un vettore di k (parametro) componenti con valore in [0, 1]: 1 se x, il dato in input, coincide con u^(i) (parametri). Più x è distante da u^(i), più il valore si avvicina a zero secondo la relativa distribuzione a campana con ampiezza _gamma_ (parametro). Più aumenta k, più il modello diventa complesso. Più aumenta gamma, più il modello diventa semplice.

# Recommendation

I sistemi di raccomandanzione mirano ad individuare relazioni tra utenti e prodotti. Esistono macro famiglie di approcci sistematici: collaborativi, content-based, knowledge-based e ibridi.  
La raccomandazione deve confrontarsi con il fenomeno della long tail, ossia nel mondo online pochi prodotti sono acquistati/visionati molto e molti prodotti sono acquistati/visionati poco.

## Matrice di utilità

In un Recommender System agiscono due attori principalei, utenti e prodotti. I dati sono organizzati in una matrice, detta Utility Matrix o matrice di rating in cui ogni riga corrisponde ad un utente e ogni colonna ad un prodotto. Ogni cella corrisponde alla valutazione dell'untente per il prodotto.  
La matrica può essere binaria (comprato/non) oppure discreta (valutazioni). La matrice è generalmente molto sparsa.

## Tipi di giudizi

### Giudizi espliciti

I giudizi espliciti rappresentano i feedback con maggiore probabilità di precisione. Sono i più comunemente utilizzati.  
Il problema principale è che gli utenti non sempre sono disposti a votare molti prodotti.

### Giudizi impliciti

I giudizi impliciti sono tipicamente raccolta da applicazioni in cui il sistema di raccomandazione è incorporato.  
Possono essere raccolti costantemente e non richiedono ulteriori sfrozi dal lato dell'utente, il loro problema è che possono essere ambigui.

## Misurare la bontà della raccomandazione

Esistono diverse metriche di misurazione dell'error rate:
- MAE, Mean Absolut Error: calcola la deviazione tra giudizi previsti e reali. Maggiore è la deviazione minore è l'accuratezza della raccomandazione
- RMSE, Root Mean Squared Error: simile al MAE, ma pone maggiormente l'accento sulla maggiore deviazione

## Sistemi collaborativi

I sistemi collaborativi sono un insieme di approcci chiamati di Collaborative Filtering (CF).  
Il loro senso è quello di fare recommendation utilizzando la saggezza della massa. L'assunzione di base sta nel fatto che siano disponibili i voti degli utenti per gli articoli del catalogo. L'ipotesi  che i clienti che hanno avuto interessi simili in passato avranno interessi simili in futuro.  
Esistono due approcci classici:
- Memory-based: utilizzano direttamente i dati per rilevare correlazioni tra utenti (o elementi) e raccomandare ad un utente un oggetto che non ha acquistato/valutato
- Model-based: l'obiettivo è il medesimo ma utilizzano algoritmi di apprendimento automatico per la creazione di modelli di learning per fare recommendation ad ogni utente

In input si ha l'utility matrix e in output una previsione di quanto l'utente gradirà o meno un determinato prodotto.

### User-based nearest-neighbor CF

Questa CF si basa sull'ipotesi che se gli utenti in passato hanno votato/acquistato in maniera simile, lo faranno anche in futuro

## Misurare la similarità tra utenti

### Correlazione di Pearson

Si parte dall'assunzione che le variabili hanno distribuzione gaussiana. Il risultato è compreso tra -1 e 1: se è maggiore di 0 la correlazione è positiva, se è 0 non c'è correlazione e se è minore di 0 c'è correlazione inversa.  
Per correlazioni non lineari e variabili non guassiane si utilizza Spearman.

### Similarità coseno

Il risultato ha valori compresi tra 0, massima dissimilarità, e 1, massima similarità.

## Migliorare le metriche di predizione

Non tutte le valutazioni degli utenti simili sono ugualmente utili. Una possibile soluzione è quella di dare maggior peso agli elementi sui quali esiste maggiore varianza sui relativi voti. Ha quindi molto importanza il numero di elementi co-rated. Si può anche amplificare il peso degli utenti "molto simili", cioè con un valore di similarità vicino ad 1.  

## Approcci memory e model based

Il CF user-based è considerato basato sulla memoria poiché la matrice di valutazione viene direttamente utilizzata per trovare gli utenti simili e per fare previsioni. Questo approccio può non scale. Per questo è meglio utilizzare un approccio model based, cioè basati su una fase di pre-elaborazione o model-learning: il modello addestrato è poi usato per fare previsioni.

### Item-based CF

Un approccio model-based è quello item-based CF. L'idea di base è quella di utilizzare la similarità tra prodotti (e non utenti) per fare prevedere quale voto darebbe l'utente u al prodotto p. È dimostrato che il calolcolo della similarità tramite coseno in questo caso funziona meglio.  

L'approccio item-based non risolve il problema della scalabilità anche se le similarità tra prodotti dovrebbero essere più stabili di quelle tra utenti  
Un altro problema è quello del cold start: come si possono raccomandare nuovi elementi, cosa si può consigliare ai nuovi utenti? Esistono vari approcci per risolvere il problema:
- Diretti: chiedere/forzare gli utenti sulla valutazione, utilizzare un metodo basato sulla popolarità generale dei prodotti o semplicemente non raccomandare nulla
- Algoritmi specifici: assumendo una transitività dei simili si può utilizzare il nearest-neighbor
- Approccio a dati sparsi

### Metodi Graph-based

Spreading activation sfrutta la transitività presunta dei gestu del client e, quindi, aumenta la matrice con informazioni aggiuntive.  
Sfruttando più in profondo la transitività, si possono utilizzare associazioni indirette con percorsi più lunghi. Cioè utilizzando ad esempio una transitività a 5 passi piuttosto che a 3.  

### Altri metodi model-based

Negli ultimi anni sono state proposte numerose tecniche:
- Tecniche statistiche di matrix factorization come SVD
- Regole associative
- Modelli probabilistici, modelli di clustering, reti bayesiane, Latent Semantic Analysis probabilistico
- Approcci ad apprendimento automatico più complessi

### SVD Singular Value Decomposition

L'idea di base è quella di creare modelli più complessi offline per la produzione di previsioni più veloci. Si utilizza SVD per la riduzione della dimensionalità delle matrici di rating.  
SVG genera un nuovo spazio con nuovi assi dove colloca i dati della matrice; i nuovi assi rappresentano le dimensioni lungo le quali i dati variano maggiormente, tuttavia non sono sempre facilmente inrepretabili; cattura i segnali rilevanti nei dati filtrando il rumore con un numero k di dimensioni molto inferiore alle dimensioni originali (k compreso tra 20 e 100).  
Le raccomandazioni utilizzando SVD sono fornite in tempo costante.  

SVD afferma che una Matrice M può essere fattorizzata nel prodotto di 3 matrici, dove la prima contiene la similarità tra utenti, la seconda è una matrice diagonale con gli autovalori (ognuno rappresenta la varianza dei dati sulla dimensione che rappresenta) e gli assi della terza rappresentano le variabili latenti.

Quindi, con la matrix factorization SVD si individuano i fattori latenti (ossia le dimensioni del nuovo spazio), si generano approssimazioni di matrici a basso rango k e si ha la proiezione di oggetti e utenti nello stesso spazio n-dimensionale.  

I parametri possono essere determinati e ottimizzati solo su esperimenti in un determinato dominio, perciò l'accuratezza delle raccomandazioni può anche diminuire rispetto all'impiego della matrice di rating originale con valori di k inadeguati.

### Recommendation con regole associative

Le regole associative si ricavano dall'analisi delle frequenze delle combinazioni di acquisto dall'intero insieme di transazioni di acquisto. Le regole associative sono implicazioni statistiche del tipo x->y.  
Esistono due misure di accuratezza delle regole associative, utilizzate anche come cut-off per scegliere le regole migliori: supporto, numero di transazioni dove x è acquistato insieme ad y diviso il numero totale di transazioni, e la confidenza che ha lo stesso numeratore del supporto e a denominatore il numero delle tansazioni che contengono x.

### Metodi probabilistici

L'idea di base è "Qual è la probabilità che un utente esprima un rating r di un prodotto p dati i rating dell'utente e di tutti gli altri utenti?".

### Slope One Predictor

Slope One Predictor è semplice e si basa su un differenziale di popolarità tra gli elementi per gli utenti. Lo schema di base è quello di prendere la media di queste differenze dei co-rating per fare la previsione.  
In generale bisogna trovare una funzione della forma f(x) = x + b.  
Se un utente ha votato diversi elementi, le previsioni possono essere combinate utilizzando una media ponderata. Una buona scelta per il peso è il numero di utenti che hanno valutato entrambi gli elementi.

### Riassunto metodi di collaborative filtering

Pro: facile da comprendere e implmenetare, funziona bene in alcuni settori.
Contro: richiede comunità di utenti, ha il problema della sparsità e non c'è nessuna integrazione di altre fonti di conoscenza.  
Il metodo più utilizzato per valutare un reccommender system è il RMSE.

# Classificazione

Classificare significa individuare uan funzione che massimizzi la separazione tra le classi.

## Classificazione lineare con iperpiani

Sono metodi che assumono l'esistenza di separazioni lineari dei dati. Consistono nell'individuazione di iperpiani di separazione dell classi con programmazione lineare oppure con soluzioni iterative, come ad esmepio perceptron, newton, discesa del gradiente.  
Un iperpiano w * x + b = 0 partiziona lo spazio in 2 parti ed è definito dai 2 parametri w, vettore unitario perpendicolare all'iperpiano di separazione, e b, distanza ortogonale dell'iperpiano dall'origine detta anche intercetta.  
In 2D un iperpiano è una retta.

### Alcune caratteristiche dei classificatori

Esistono molte possibili soluzioni con parametri distinti w e b. Alcuni metodi individuano un iperpiano di separazione non ottimale, in base ad un qualche criterio di ottimalità, altri individuano un iperpiano di separazione ottimale.  
Tutti i dati influenzano la ricerca di un iperpiano nella maggiorparte dei metodi (perceptron, regressione logistica, naive bayes...) mentre il support vector machines è influenzato solo dai punti difficili, i c.d. decision boundary.  
Esistono poi problemi con dati non separabili linearmente, risolvibili con soluzioni che trasformano lo spazio dei dati in modo che le classi diventino separabili linearmente (svm, reti neurali...) o soluzioni intrinsecamente non lineari (kNN, decision tree, randomforest, xgboost...).  
In generale, più variabili hanno i dati, più chance ci sono di separare le classi linearmente.

### Perceptron

Perceptron è il progenitore delle reti neurali. Converge solo se i dati sono separabili linearmente.  
Come possiamo trovare l'iperpiano di separazione? Dato che ogni istanza è etichettata con -1 o +1, possiamo minimizzare l'errore sulle m istanze di training, cercando il massimo della funzione max(0, -y(b+w*x) <0) che restituisce 0 quando l'istanza è correttamente classificata.  
La funzione suddetta è però continua e convessa ma non derivabile per questo occorre utilizzare la logistic loss se serve la discesa del gradiente.

#### Logistic loss

Sostituiamo la funzione di prima max(0, s) con una funzione derivabile. Una sua approssimazione è nota come softmax(0, s) = log(1+e^s) che è convessa, continua e derivabile. Perciò minimizzando la somma rispetto a b e w della softmax su tutte le istanze, si minimizza l'errore.  
Questa formulazione è conosciuta come logistic loss a cui si può aggiungere la regolarizzazione L2 (L1) dei parametri w con peso _lambda_.

### Regressione logistica o sigmoide

La regressione logistica deriva dalla logistic loss con norma L2 in forma compatta (senza b). PUò essere considerata la versione moderna del Perceptron.  
Il risultato è interpretabile come probabilità di appartenenza di ogni istanza t ad una delle classi. Approssima una funzione a gradino.

#### Regressione logistica in due classi: multivariata lineare e non lineare

Vedere <a href="./Riassunto.md">il riassunto principale </a>.

## Classificazione multiclasse e iperpiani

Esistono due metodi con C classi:
- One-Versus-All: si individuano C iperpiani, uno per ogni classe da separare da tutte le altre, con lo stesso metodo di individuazione di un singolo iperpiano. È parallelizzabile.  
  La regola di fusione è la seguente: ad ogni istanza x si assegna la classe y corrispondente al piano j che massimizza
- Multinomial: individua congiuntamente C iperpiani minimizzando la regola di fusione di cui sopra

## Classificazione con classi sbilanciate

In molti problemi reali la suddivisione di instanze tra classi è molto sbilanciata, di conseguenza si hanno molti più errori di classificazione sulla classe meno rappresentata, spesso con risultati inaccettabili.  
Una possibile soluzione è quella di aumentare il peso degli errori sulla classe con meno istanze.

## Individuazione dell'iperpiano

L'iperpiano individuato da Perceptron e Regressione logistica è uno dei tanti possibile iperpiani di separazione. Come definiamo il concetto di migliore?  
Il numero di soluzioni diminuisce se cerchiamo una separazione lineare con il maggiore margine possibile tra le istanze delle due classi, possiamo quindi individuare l'iperpiano che massimizzi il margine (minore overfitting).  
L'iperpiano migliore è definito dai punti difficili chiamati support vectors, cioè tutti quei punti vicini al decision boundary. Quindi il problema diventa risolubile come problema di ottimizzazione quadratica attraverso l'utilizzo delle Support Vector Machines.

## Valutazione di modelli di classificazione

I modelli di classificazione si valutano sul test/validation set calcolando una matrice ClasseReale x ClassePredetta.

### Classificazione e tasso di errore

Il tasso di errore calcolato sul training set è inevitabilmente ottimistico rispetto all'errore atteso su nuovi dati. Per questo, i dati nei problemi reali sono suddivisi in tre subset: training, validation (per il tuning degli iperparametri) e test (per simulare il tasso di errore sui nuovi dati).  
Supponiamo, ora, che un classificatore abbia un tasso di successo sul test set del 75%. Quanto è attendibile questa accuratezza sull'intera popolazione dei dati? La risposta non sarà un singolo valore, piuttosto un intervallo: 75% +- %err.  
In ogni progetto di data science con modelli predittivi, queste misure sono essenziali per valutare l'affidabilità del risultato.

### Modellazione della classificazione come Processo di Bernoulli

La classificazione di N istanze è modellabile con un processo Bernoloulliano di N eventi binari indipendenti: errore o successo.  
Siano N gli esperimenti, S i successi. Si ha f = S/N --> tasso di successo (accuratezza).

# Natural Language Processing

## Dati strutturati e destrutturati

I dati strutturati sono tutti quei dati provvisti di un modello, o schema, capace di descriverli e attribuirgli semantica.  
I dati destrutturati sono invece tutti quei dati sprovvisti di un modello in grado di spiegarne formalmente la semantica.  

Il dato destrutturato di maggior interesse è il testo in linguaggio naturale. Il testo segue regole linguistiche come sintassi, lessico, grammatica, ma in generale è ambiguo e dipendente dal contesto, senza considerare la possibile presenza di errori e l'utilizzo costante di elementi non previsti dalla lingua.

## Pre-processing di un testo

Del testo è inizialmente disponibile come una sequenza di caratteri in forma non strutturata.

### Segmentazione (Tokenization)

La prima operazione compiuta in genere sul testo è la sua segmentazione in elementi sintattici di base (token). L'operazione più comune consiste nello scomporre il testo nelle sue singole parole (word tokenization). Testi non brevi possono anche essere dapprima divisi in frasi (sentence tokenization).

#### Word tokenization

Per una segmentazione accurata è utile integrare regole e modelli specifici della lingua analizzata.

#### Part of Speech POS

Le POS sono le classi grammaticali a cui ciascun elemento di una frase può appartenere (nome, verbo, aggettivo...). Esistono diverse tassonomie di POS, da utilizzare a seconda delle esigenze.  

Il POS tagging consiste nell'etichettare ciascuna parola estratta dalla segmentazione del testo col suo POS nella frase. È utile ad esempio se si vogliono filtrare parole di tipo specifico o se elaborazioni successive dipendono dai POS mentre non è necessario ad esempio se si vogliono estrarre le parole chiave del documento indipendentemente dalla loro posizione nella frase.  

La sequenza delle parole estratte va spesso ripulita prima di essere utilizzata nelle analisi successive poiché alcuni elementi estratti possono non essere necessari e altri elementi possono essere "normalizzati".

#### Pulizia

##### Casefolding

Il casefolding consiste nel convertire tutte le parole completamente in minuscolo, è un metodo basilare per uniformare token.

##### Rimozione stopword

Articoli, preposizioni e congiunzioni non danno indicazioni sull'argomento. Queste parole, dette stopword, sono spesso rimosse dalla rappresentazione di ciascun documento.

##### Lemmatizzazione

La lemmatizzazione consiste nel sostituire ciascuna parola col suo lemma, ovvero la sua forma base presente nel dizionario.

##### Stemming

Diversamente dalla lemmatizzazione, lo stemming ricava da ogni parola la sua radice morfologica. La radice può non essere una parola di sensio compiuto mentre il lemma lo è.  
Lo stemming si usa come alternativa più efficiente alla lemmatizzazione per unificare termini simili. Ovviamente lo stemming è più approssimativo ed il suo utilizzo può comportare una perdita di precisione.

## Full Text Search

L'Information Retrieval si occupa della ricerca di testo in documenti, ossia in testo non strutturato. Ciò richiede l'estrazione e l'indicizzazione dei termini in ogni documento.  
Si possono volere individuare anche termini simili a quelli estratti esplicitamente.  
Per soddisfare al meglio la query dell'utente i documenti individuati devono essere elencati a partire dal più rilevante e la ricerca deve essere il più efficiente possibile.

### Compromesso tra esattezza e completezza

Data una o più query di test, è possibile misurare la bontà del mistema in termini di precision, percentuale di documenti rilevanti tra quelli restituiti, e recall, percentuale di documenti rilevanti restituiti tra tutti quelli rilevanti nella collezione.  
Tendenzialmente, cambiando i parametri di un sistema FTS per migliorare uno dei due aspetti, l'altro peggiora.

### Modeli

La disciplina dell'Information Retrieval ha elaborato diversi modelli per rappresentare e recuperare documenti.

### Modello booleano standard

Nel modello booleano, i documenti sono semplicemente divisi tra rilevanti e non, senza un ordine preciso. Ad un generico termine di ricerca viene associato un insieme di documenti rilevanti, quelli che contengono il termine.  
I termini possono essere combinato con operatori logici: A & B (congiunzione logica, intersezione tra gli insiemi di documenti rilevanti) e A | B (disgiunzione logica, unione tra gli insiemi).  
Il pro di questo modello è che è concettualmente semplice e facile da implementare. Di contro ha che restituisce i documenti senza un ordine significativo e spesso in quantità troppo alta o troppo bassa.

### Estrarre la semantica di un test

Una comprensione completa ed esatta del significato di un testo è molto complessa per un calcolatore. D'altra parte però la comprensione esatta è spesso non necessaria.   
Per capire il significato di un testo, è necessario conoscere le parole esistenti e come queste siano in relazione tra loro, per questo si usano delle basi di conoscenza esterne del linguaggio.

### WordNet

WordNet è tra i più noti database lessicali della lingua inglese.  
Le principali relazioni semantiche in WordNet sono: iponimia (is-a), meronimia (essere parte di), implicazione e antonimia (essere opposto di).

### Word Sense Disambiguation

La WSD è la procedura che associa ad ogni parola in un testo il suo significato esatto. La WSD richiede una base di conoscenza che associ ad ogni parola i significati possibili.  
La disambiguazione avviene in genere in base al contesto di ciascuna parola, ovvero le parole vicine ad essa.

### Named Entity Recognition

Una named entity è una specifica entità a cui ci si può riferire per nome in un testo. La named entity recognition consiste nell'individuare le named entity in un testo e classificarle per tipo.

### Rappresentare il contenuto di un testo

Nell'analisi di testi serve spesso rappresentare il contenuto generale di un documento in modo strutturato e compatto. Le parole contenute in un documento sono in genere indicative del suo contenuto.

#### Bag of Words

Una Bag of Words è una rappresentazione in forma di multiset delle parole contenute in un'unità di testo (un documento). Si tratta di un elenco delle parole distinte presenti nel testo, con associata a ciascuna il numero di occorrenze in esso.

#### n-gram

Gli n-gram sono sequenze di n parole presenti in un testo. Se n=2 si parla di bigram, se n=3 si parla di trigram.  
Possono rendere più completo il BOW di un testo, includendo termini composti di più parole. Sono però inclusi anche molti n-gram non significativi e il numero totale è molto alto.

#### Rappresentazione vettoriale

Si consideri di definire un insieme D (dizionario) di termini distinti che si potrebbero trovare in un documento di testo. Il BOW di ciascun documento può essere rappresentato come un vettore di |D| elementi in cui l'i-esimo valore è il numero di occorrenze dell'i-esimo termine o in alternativa si può considerare un vettore binario con 1 per le parole presenti e 0 per le assenti.

### Vector Space Model

Il Vector Space Model prevede di rappresentare un insieme di N documenti come vettori in un medesimo spazio. Lo spazio ha un numero di dimensioni D pari al numero di parole considerate nel dizionario condiviso. L'insieme di documenti si può rappresentare in una matrice termini-documenti di dimensioni DxN in cui l'elemento i,j indica quante volte la parola i appare nel documento j.

### Term weighting

Ogni documento contiene termini più o meno significativi per caratterizzare il suo contenuto. Il numero di occrrenze costituisce un'indicazione immediata del peso di un termine all'interno di un documento, esistono però diversi schemi di term weighting per ottenere pesi più accurati sulla base di altre informazioni.

### TF-IDF e Vector Space Model

Nel VSM occorre determinare come assegnare i valori (pesi) a ciascun termine in ciascun vettore.  
TF, term frequency, pesa la rilevanza di ciascun termine in ciascuno dei singoli documenti.  
IDF, inverse document frequency, pesa l'importanza di ciascun termine nell'intera collezione. È quindi più alto per termini che compaiono in meno documenti.  
Il TF-IDF per un termine t e un documento d è il prodotto dei logaritmi (log perché l'importanza dei termini non cresce linearmente con la loro frequenza) di TF e IDF.

### Normalizzazione dei vettori

Per garantire pari peso a documenti di lunghezze diverse, si usa normalizzare i vettori ottenuti in seguito al term weighting.  
La normalizzazione mantiene la direzione del vettore, che indica la frequenza relativa tra le parole e quindi l'argomento.

### Similarità coseno

Rappresentando i documenti come vettori nel medesimo spazio, possiamo misurarne la similarità con diverse metriche. Per confrontare coppie di documenti si usa spesso la similarità coseno, pari al coseno dell'angolo tra i vettori.  
Per vettori con tutti valori positivi, la similarità coseno è sempre inclusa tra 0 (angolo 90°, i documenti non hanno parole in comune) e 1 (angolo 0°, i vettori hanno la stessa direzione).

### VSM vs modello booleano

Esistono modelli estesi che combinano VSM e logica booleana: il modello booleano esteso prima recupera i documenti che soddisfano la proposizione booleana, poi il risultato è ordinato calcolando i rank dei documenti. Quindi il modello esteso aumenta il potere espressivo del VSM con proposizione booleane.

### Modeli con dipendenze tra termini

Nei modelli visti finora, ciascun termine è considerato a se stante e indipendente da tutti gli altri. In altri modelli sono considerate esplicitamente dipendenze reciproche tra termini.  
La Latent Semantic Analysis traspone documenti e query in un nuovo spazio dove le dimensioni corrispondono idealmente a concetti semantici, ciascuno dato da un autovettore nello spazio originale.

### Ricerca di un termine

Per ottenere la lista di documenti rilevanti ad una query, bisogna individuare in quali documenti compaiono i termini ricercati. La soluzione è una ricerca sequenziale su tutti i documenti, questo però comporta necessariamente tempi lunghi.

#### Indicizzazione

Per ricerche più efficienti si utilizzano degli indici.  
Una struttura tipica è l'indice inverso (inverted index) che associa ad ogni termine la lista di documenti in cui compare. Dato un termine, l'accesso avviene in un tempo minimo e costante O(1) alla lista di documenti.

# Machine Learning e classificazione

Le tecniche di machine learning permettono di apprende conoscenza dai dati per effettuare predizioni su di essi.

## Algoritmo di apprendimento supervisionato

Esiste una grande varietà di algoritmi di apprendimento, che generano diversi tipi di modelli di conoscenza:
- modelli probabilistici: calcolano la classe di appartenenza più probabile in base alle caratteristiche dell'oggetto
- k-nearest neighbor: verificano quale sia la classe più ricorrente tra i k oggetti del training set con caratteristiche più simile a quello dato
- altri come alberi decisionali, SVM, reti neurali...

È necessario definire le caratteristiche (feature) degli oggetti tramite un'adeguata rappresentazione strutturata. Una soluzione tipica è rappresentare ciascun oggetto come un vettore in un unico spazio multidimensionale.

Col machine learning è possibile estrarre un classificatore da recensioni per-etichettate come positive e negative. Il modello ottenuto potrà essere usato per stimare la polarità di altre recensioni dello stesso contesto.  
Per trattare i testi, è necessario rappresentarli come vettori: possiamo usare il Vector Space Model.  
La presenza di ciascun termine costituisce una feature dei testi. Per confrontare due testi tra loro si può usare la similarità coseno.
