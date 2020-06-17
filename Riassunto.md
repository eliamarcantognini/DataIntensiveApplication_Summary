# Regressione Lineare

## Modello di conoscenza

Funzione che associa ad ogni dato in input una classe/valore numerico

## Processo di estrazione della conoscenza - Fasi del learning supervisionato

- Raccolta dei dati, analisi esplorativa e della loro qualità
- Trattamento di valori mancanti, normalizzazione, standardizzazione
- Selezione di feature, ossia variabili rilevanti rispetto agli obiettivi
- Divisione dei dati in training set, validation set e test set
- Estrazione di modelli di conoscenza dal training set con algoritmi di Machine Learning
  - Individuando i parametri che massimizzino l'accuratezza sul validation set
  - Il test set misura l'accuratezza sui dati nuovi/ignoti a regime
- Depolyment della conoscenza in applicazioni

## Regressione Lineare - predire variabili continue

La regressione lineare stima da un data set una funzione lineare che associ variabili _indipendente_ di input (domini) ad una variabile _dipendente_ di output (codominio).  
Se si ha una sola variabile indipendente si parla di regressione univariata, con più variabili indipendenti si ha invece la regressione multivariata.  
La funzione ha una serie di _parametri_ i cui valori devono essere determinati in modo:
-**diretto**: soluzione ottima ma solo per problemi convessi, costo O(N<sup>3</sup>) e intera allocazione in memoria; inadeguato per ampi dataset, impossibile per big data
-**numerico**: si usa la discesa sul gradiente, che non garantisce soluzioni ottime ma è parallelizzabile, incrementale e anche per problemi non convessi

### Regressione Lineare Univariata

La regressione univariata consente di stimare una variabile diepdente _y_ sulla base di un'unica variabile indipendente _x_. In un modello di regressione lineare si assume che _y_ vari in modo direttamente proporzionale ad _x_. Ci si aspetta quindi una funzione nella forma tipica della retta, _y=ax+b_.  
L'obiettivo dell'analisi di regressione è trovare i valori dei parametri _a_ e _b_ che rendano la stima più accurata possibile.
Non esiste una combinazione dei due parametri che sia "perfetta", ma dobbiamo individuare i valori che diano la migliore approssimazione possibile.  
Denotiamo con θ la combinazione di valori dei parametri incogniti.

### Misura dell'errore

Come si può misurare l'errore complessivo sull'insieme di predizioni? La più comune formula è quella della _media dei quadrati_ dei singoli errori, **Mean Squared Error**.  
Per ottimizzare l'accuratezza della predizione, dobbiamo trovare i parametri per cui l'errore sia il minimo possibile.  
Fissato un set di dati, l'errore è calcolato come una _funzione continua sui parametri_ del modello di predizione.  
L'obiettivo della regressione è quindi quello di trovare i parametri per cui il valore della funzione d'errore sia minimo.

### Gradiente di una funzione

Data una funzione _f_ a pi variabili, la **derivata parziale** su una di esse è la derivata di _f_ considerando le altre come costanti.
Il **gradiente** è il _vettore delle derivate parziali_ della funzione _f_ per ciascuna delle sue variabili.
Intuitivamente, il gradiente calcolato in un punto **x** indica l'_inclinazione_ della curva nel punto stesso.

#### Discesa del Gradiente

La **discesa del gradiente** è un metodo iterativo per trovare un _minimo locale_ di una funzione "seguendo" il suo gradiente:


1. Si parte da un punto x<sub>k</sub> = x<sub>0</sub> a caso
2. SI valuta la funzione _f_ ed il suo gradiente nel punto x<sub>k</sub>
3. Ad x<sub>k</sub> si _sottrae un vettore proporzionale al gradiente_ per ottenere un punto x<sub>k+1</sub> con _f(x<sub>k+1</sub>) < f(x<sub>k</sub>)_
4. Si pone _k_ <-- _k+1_ e si esegue l'iterazione successiva dal punto 2, ripetendo fino alla **convergenza ad un minimo**  

La lunghezza del passo di discesa (_step size_), cioè il rapporto tra gradiente e spostamento ad ogni passo, è un **iperparametro** dell'algoritmo di discesa da impostare. E' noto nelle reti neurali come _learning rate_.  

Applicando quindi la discesa del gradiente sulla funzione d'errore, possiamo trovare i valori dei parametri per cui è minima.

Con la regressione, possiamo quindi usare i dati disponibili per ottenere in automatico una funzione che predica la nostra variabile continua:
-   Basta fornire i dati disponibili in input all'algoritmo
-   Si può riaddestrare periodicamente su dati aggiornati e _misurarne l'accuratezza_ prima di farne il deployment
- In base alla natura dei dati ed al modello di regressione scelto, la previsione della regressione può migliorare le stime fatte da esperti

### Regressione Lineare Multivariata

Con la regressione ***multivariata*** si stima una variabile dipendente _y_ sulla base di più variabili indipendenti _x<sub>1</sub>,...,x<sub>n</sub>_. La previsione avviene a partire da un numero arbitrario di variabili numeriche e categoriche
- Ogni variabile categorica si converte in più variabili **binarie** con valore 0 o 1, creandone tante quante sono i valori della variabile categorica

Nel caso della regressione lineare, la forma della funzione dei parametri incogniti stimata è un iperpiano in uno spazio a _n+1_ dimensioni.

Nella regressione lineare multivariata, si considera la variabile obiettivo come una _somma pesata delle sue variabili indipendenti_:
- La regressione determina il peso di ogni variabile numerica ed ogni variabile categorica
- La variabile obiettivo è la somma delle variabili numeriche note moltiplicate per i rispettivi pesi dei parametri, compresi quelli categorici

Il modello generale di regressione lineare multivariata è:
$$ h_0(x_1,...,x_n) = \theta_0+\theta_1x_1+...+\theta_nx_n $$

Come già visto, possiamo valutare l'errore di un modello come media dei residui quadrati.

## Normalizzazione dei dati

In generale, le variabili coinvolte in un modello di regressione possono utilizzare scale di valori molto diverse, questo rende difficile la convergenza della discesa del gradiente.  
Una soluzione consisterebbe nell'usare lunghezze di passo differenziate per parametro ma una soluzione più semplice è però normalizzare i dati in modo che tutte le variabili abbiano valori in un medesimo intervallo.  
Possiamo eseguire la discesa del gradiente su dati normalizzati e "denormalizzare" i parametri ottenuti.
Per addestrare efficacemente un modello di regressione, è comune normalizzare tutte le variabili, anche la dipendente.  
Addestrando un modello su dati normalizzati, i suoi parametri saranno ittimizzati su di essi. Per usare il modello occorre quindi coerentemente normalizzare gli input e denormalizzare gli output ottenuti.

## Valutazione del Modello di Regressione

Un modello di regressione è addestrato in modo da minimizzare l'errore su un insieme di dati noti e il suo obiettivo è ottenere un modello generale, cioè capace di fare previsioni sui nuovi dati con la stessa accuratezza mostrata sui dati impiegati per ottenere il modello stesso. Se invece l'accuratezza su nuovi dati è significativamente più bassa allora ci possono essere due problemi:
- **overfitting**: il modello è troppo aderente ai dati di addestramento perdendo di generalità, serve quindi ridurre il numero di variabili o _regolarizzare il modello_
- il training set impiegato non è statisticamente rappresentativo delle peculiarità dei nuovi dati

Per scongiurare questi problemi occorre verificare/validare l'errore del modello su dati _non_ usati per addestrarlo

### Validazione con metodo Hold-Out

Il metodo _hold-out_ prevede di dividere i dati a disposizione in due insieme, secondo una proporzione predefinita, solitamente 70-30:
- **Training set**: utilizzato per addestrare il modello di regressione, minimizzando l'errore su di esso
- **Validation set**: usato dopo l'addestramento per verificare l'errore del modello su dati ignoti

Se l'errore sul validation set è simile a quello sul training set, si assume che il modello abbia generalizzato bene i dati.

### Errore relativo

L'errore quadratico medio è funzionale all'addestramento del modello di regressione tramite discesa gradiente, possiamo in aggiunta valutare l'errore secondo altri criteri, ad esempio l'_errore relativo_ tra un valore _y reale_ e la _stima y_ è definito da:
$$ \frac{|y_p-y_r|}{y_r}$$

### Coefficiente di Determinazione R<sup>2</sup>

Il coefficiente di determinazione R<sup>2</sup> indica la proporzione tra variabilità dei dati e corretteza del modello.  
Il suo valore è compreso tra:
- 0, indica che non c'è alcuna relazione tra modello e dati
- 1, indica che il modello descrive perfettamente i dati

# K-Fold cross validation

È un metodo alternativo al metodo di hold-out per la divisione del dataset in training e validation set. Consiste in:
- Suddividere i dati in k sottoinsiemi disgiunti
- Un sottoinsieme è usato come validation set e gli altri k-1 come training set
- Si ripete lo stesso procedimento k volte con ciascuno dei k subset
- Si ottengono k modelli di learning, e.g. k regressioni
- L'accuratezza è la media delle accuratezza dei k modelli

Esiste anche la **k-fold cross validation stratificata**, che è come la normale k-fold cs ma con stessa distribuzione e caratteristiche dei dati in ogni fold.

# Nested Cross Validation

È una versione "migliorata" della K-Fold cross validation.  
Ogni parte di training della k-fold cs esterna è suddivisa nella cross validation intera in m-subfold che sono usati per indivudare gli iperparametri migliori.  
Gli iperparametri migliori sono poi usati per addestrare e testare il modello nella realtiva parte della validation esterna.  
Non è un metodo per estrarre un modello predittivo migliore di un altro, ma per stimare l'accuratezza a regime sui dati.

# Regressione Non Lineare

## Regressione Polinomiale

La _regressione polinomiale_ è una generalizzazione di quella lineare con altri termini di grado superiore, utile per ottenere modelli capaci di descrivere data set più complessi.  
La regressione polinomiale è ancora lineare, rispetto ai parametri θ, non nelle variabili dei dati (a, b) che però sono noti.  

### Complessità del modello

Ricordiamo che i dati devono essere divisi in training e validation set e qualsiasi modello di learning si estrae al training set.  
All'aumentare della complessità del modello di learning si riduce l'errore, ma dopo una certa soglia di complessità l'errore sul validation set torna a cresce! Questa è la soglia di _overfitting_ che indica che la maggior complessità dei modelli descrive meglio il training set ma non il validation set, cioè **il modello ha perso di generalità**.  
Dobbiamo quindi scegliere il modello che minimizzi l'errore sul validation set.  
Il modello può essere:
- **overfitting**: il modello è troppo complesso, l'errore sul training set è significativamente inferiore a quello sul validation, il modello non è generale
- **underfitting**: il modello è troppo semplice e quindi inadeguato a rappresentare i dati, è insoddisfacente sia l'errore sul training che sul validation
- **optimal**: minimizza l'errore sul validation set

Dal training set si determinano i parametri θ del modello di learning, e.g. i coefficienti α e β nella regressione, mentre gli iperparametri sono tutte le altre scelte e si determinano dal validation set. Esempi di iperparametri sono il tipo di funzione di regressione (lineare o polinomiale), il grado, la normalizzazione, la regolarizzazione, etc..

### Regolarizzazione

Il grado nella regressione polinomiale misura la complessità del modello di learning, con grado teoricamente infinito possiamo modellare qualsiasi dataset, ma i coefficienti del polinomio, e.g. grado 50, diventano molto grandi e ciò provoca forti oscillazioni nella regressione peggiorandone l'accuratezza.  
Regolarizzare significa ridurre il valore dei coefficienti!

#### Regressione Ridge

Nella regressione ridge, si aggiunge alla funzione d'errore da minimizzare anche λ*θ. λ è un iperparametro, da ottimizzare sempre rispetto al validation set, compreso tra 0 e infinito ed è l'indice di regolarizzazione: con λ pari a 0 non si ha regolarizzazione mentre più λ è grande, più si riducono i coefficienti θ.  
Il metodo più semplice per determinare il valore di λ è utilizzare la k-fold cross validation.  
Determinare gli iperparametri migliori con gli stessi dati usati per estrarre il modello migliore genera score ottimisti, per questo è meglio usare la _Nested Cross Validation_ per stimare correttamente anche lo score su dati ignoti.
La regressione Ridge utilizza la norma L2 per minimizzare l'errore.

#### Collinearità: dipendenze tra variabili di input

Nella regressione ordinaria si assume che le variabili di input siano indipendenti tra loro poiché con dipendenze la regressione sarebbe instabile.  
Per esempio, piccole variazioni nei dati generano modelli molto diversi e inaffidabili.  
Se due variabili sono in un qualche modo dipendenti l'una dall'altra, si ha un effetto noto nel _Statistical Learning_ come _interazione_. Si può risolvere questo problema aggiungendo un nuovo parametro θ che moltiplica le due variabili, ad esempio θ*x<sub>1</sub>*x<sub>2</sub>.
La regolarizzazione risolve il problema poiché vincola/riduce le possibili soluzioni.

#### Regressione LASSO

La regressione Ridge, all'aumentare di λ riduce ma non azzera il valore dei parametri θ, perciò la soluzione utilizza tutte le variabili, anche quelle irrilevanti per la predizione, dando una _soluzione densa_.
Se penalizziamo i θ nella minimizzazione dell'errore con norma L1 invece che L2, la soluzione θ* è vincolata all'interno di un ipercubo centrato sull'origine: maggiore è λ, più è probabile che θ* casa sugli spigoli azzerando diversi θ e si eliminano più _varriabili irrilevanti_, dando luogo a una _soluzione sparsa_. Questa è la regressione LASSO.
Il modello predittivo, eliminate le variabili ritenute irrilevanti, risulterà più interpretabile. La regressione Lasso è quindi utilizzata per selezionare le variabili da utilizzare.

#### Elastic Net

La regressione Elastic Net generalizza Ridge e LASSO con entrambe le penalizzazioni dei θ con norma L1 e L2. Introduce un secondo iperparametro α per pesare le penalizzazioni L1 e L2:
- Con α = 0 si ha la regressione Ridge
- Con α = 1 si ha la regressione LASSO
- Valori intermedi combinano le due penalizzazioni ed i loro pro e contro

### Esplosione della dimensionalità

Il numero di variabili indipedenti è quello del dataset, ma quanto parametri dobbiamo calcolare con θ grande (a.e. 10) e grado polinomiale (a.e. 2)? Nell'esempio, il risultato è 66 variabili. Si ha quindi un aumento delle dimensioni di quasi 7 volte.  
Si hanno quindi _problemi ad elevata dimensionalità_ se il numero di variabili è maggiore, o dello stesso ordine di grandezza, rispetto al numero di istanze.  
Generalizzando, quindi, quanti parametri θ si hanno con _n_ variabili e grado _g_? _(n+g g)_ termini.  
Una regola generale è quella di diminuire _g_ all'aumentare di _n_ e viceversa.
Ma esiste un modo per mappare i dati originali in un nuovo spazio ad elevata dimensionalità senza creare le relative nuove variabili?  

#### Funzioni Kernel

Sì, si può fare, utilizzando un metodo chiamato Kernel Trick che vale per ogni grado _g_ e numero _n_ di dimensioni.
Il metodo serve a portare i dati in nuovi spazi ad elevata dimensionalità senza creare nuove variabili e utilizza la formulazione:
$$ x=(x_1,...x_n),z=(z_1,...,z_n),Kernel(x, z) = (x ⋅ z + 1)^g $$
Esistono altre funzioni kernel, non lineari, che possono modellare data set complessi con distribuzioni non lineari. Anche kernel non lineari rispondo all'esigenza di non aumentare le dimensioni dei dati e dei relativi costi impossibili da sostenere per operare nello spazio ad elevata dimensionalità. L'unico limite è che possono generare modelli affetti da overfitting.  
_Proprietà delle funzioni Kernel_
- $$ K(x,y) = (x⋅y)$$
- Polinomiale
  - $$ K(x,y) = (x⋅y+\theta)^g $$
- Gaussian Radial Basis, _RBF_
  - È una funzione non lineare più complessa della polinomiale, restituisce un vettore di k (parametro) componenti con valore in [0, 1], 1 se x, il dato in input, coincide con u<sup>(i)</sup> (parametri). Più x è distante da u<sup>(i)</sup>, più il valore si avvicina a zero secondo la relativa distribuzione a campana con ampiezza σ (parametro). Più aumenta k, più il modello diventa complesso. Più aumenta gamma, più il modello diventa semplice 
  - $$ K(x,y) = e^{\gamma||x-y||^2} \\ \gamma=\frac{1}{2\sigma^2}$$
- Sigmoidale
  - $$ K(x,y) = \tanh(kx⋅y - \delta) \\ 0 \leq \theta \isin R, g \isin N, k \isin R, \sigma \isin R $$

# Recommendation

I sistemi di raccomandazione mirano ad individuare relazioni tra utenti e prodotti.  
Esistono macro famiglie di approcci:
- _Sistemi collaborativi_: "Dimmi ciò che è popolare tra gli utenti con interessi simili ai miei"
- _Sistemi content-based_: "Mostrami oggetti simili per contenuto a ciò che ho apprezzato in passato", e.g. individua libri simili per contenuto a quello che ho letto
- _Sistemi knowledge-based_: "Dimmi quello che si adatta a me in base alle mie esigenze", e.g. l'utente esprime le caratteristiche del prodotto di interesse e il reccomender cerca i prodotti che più lo soddisfano
- _Sistemi ibridi_: combinazioni delle tecniche precedenti

La raccomandanzione nel mondo fisico è in generale semplice mentre online il numero di prodotti esistenti è molto superiore al numero di prodotti che l'utente può analizzare. Questo dà vita al fenomeno chiamato _long tail_, ossia pochi prodotti sono acquistati/visionati molto e molti prodotti sono acquistati/visionati poco.  
I sistemi di raccomandazione sono un valore aggiunto sia per il cliente - trovare oggetti rilevanti, intrattenimento, aiuto nell'esplorazione delle scelte, migliorare l'insieme delle scelte - che per il provider - servizio personalizzato, aumento della fiducia e della fidelizzazione, aumento delle vendite per click, maggiore conoscenza sui clienti, promozione e persuasione.  

## Matrice di utilità

In un Reccomender System agiscono due attori principali, utenti e prodotti.  
I dati sono organizzati in una matrice, detta ***Utility Matrix*** o _matrice di rating_ in cui:
- Ogni riga corrisponde ad un utente
- Ogni colonna corrisponde ad un prodotto
- Ogni cella corrisponde alla valutazione dell'utente per il prodotto, oppure può essere vuota se l'utente non si è espresso sul prodotto.

La matrice può essere binaria - 0/1, se l'utente ha comprato o meno il prodotto - oppure discreta, a.e. valori da 1 a 5.  
La matrice è generalmente molto sparsa visto che ogni utente da voti o compra una piccolissima parte dei prodotti.

## Tipi di giudizi

### Giudizi espliciti

I giudizi espliciti rappresentano i feedback con maggiore probabilità di precisione. Sono i più comunemente utilizzati.  
Il problema principali sono che gli utenti non sempre sono disposti a votare molti prodotti.

### Giudizi impliciti

I giudizi implici sono tipicamente raccolti dal negozio web o da applicazioni in cui il sistema di raccomandazione è incorporato, a.e. l'acquisto di un prodotto è considerata una valutazione positiva, numero di click, pagine viste, tempo speso sulle pagine, etc...  
Possono essere raccolti costantemente e non richiedono ulteriori sforzi dal lato dell'utente.  
Il loro problema è che non si può essere certi che il comportamento degli utenti sia interpretato correttamente.  
La strada migliore è sicuramente quella di utilizzare i giudizi impliciti insieme a quelli esplici, con la possibilità di domandare all'utente una conferma sulla correttezza dell'informazione.

## Misurare la bontà della raccomandazione

Esistono diverse metriche di misurazione dell'error rate:
- Mean Absolut Error (_MAE_): calcola la deviazione tra giudizi previsti p<sub>i</sub> e quelli reali r<sub>i</sub>. Maggiore è la deviazione minore è l'accuratezza della raccomandazione
  - $$ MAE=\frac{1}{n}\sum^n_{i=1}|p_i-r_i| $$
- Root Mean Squared Error (_RMSE_): simile al _MAE_, ma pone maggiormente l'accento sulla maggiore deviazione
  - $$ RMSE=\sqrt{\frac{1}{n}\sum^n_{i=1}(p_i-r_i)^2} $$
  
## Sistemi collaborativi

I sistemi collaborativi sono un insieme di approcci chiamati di _Collaborative Filtering_ (CF). Sono generalmente i più utilizzati.  
Il loro senso è quello di fare recommendation utilizzando la _saggezza_ della massa. L'assunzione di base sta nel fatto che siano disponibili i voti degli utenti per gli articoli del catagolo (implicitamente o esplicitamente). L'ipotesi è che i clienti che hanno avuto interessi simili in passato, avranno interessi simili in futuro.  
Esistono due approcci classici:
- _Memory-based_: utilizzano direttamente i dati come gusti, voti, click, etc, per rilevare correazioni tra utenti (o elementi) e raccomandare ad un utente _u_ un oggetto che non ha valutato/acquistato
- _Model-based_: l'obiettivo è il medesimo ma utilizzano algoritmi di apprendimento automatico per la creazione di modelli di learning per fare reccomendation ad ogni utente
  
In input si ha l'Utility matrix e in output una previsione di quanto l'utente gradirà o meno un determinato prodotto.

### User-based nearest-neighbor CF

Considerato un utente _u_, vogliamo predire il gradimento di un prodotto _p_ che non ha votato/comprato. Dobbiamo:
- Trovare il set di utenti più simili in base ai voti sugli stessi prodotti, ma che hanno votato il prodotto _p_ non votato da _u_
- Utilizzare, ad esempio, la media dei voti assegnati da questo set di utenti al prodotto _p_ per predire il voto di _u_
- Si ripete questa operazione per tutti i prodotti non ancora votati da _u_ e si raccomanda quello con voto maggiore

Questa CF si basa sull'ipotesi che se gli utenti in passato hanno votato/acquistato in maniera simile, lo faranno anche in futuro.

### Misurare la similarità tra utenti

#### Correlazione di Pearson

Si parte dall'assuzione che le variabili hanno distribuzione gaussiana. I dati:
- a, b: sono due utenti
- r<sub>a,p</sub> e r<sub>b,p</sub>: rispettivamente rating dell'utente a e dell'utente b per il prodotto p
- P: set di prodotti per cui sia a che b hanno espresso un giudizio
- r<sub>a</sub> barrato e r<sub>b</sub> barrato: valore medio rispettivamente dei giudizi dell'utente a e b per i prodotti in P
Il risultato è compreso tra -1 e 1:
- Se è maggiore di 0, la correlazione tra le due variabili è positiva
- Se è 0, non c'è correlazione
- Se è minore di zero, c'è correlazione inversa tra le variabili

La formulazione è la seguente:
$$ sim\_pearson(a,b) =
 \frac
{ \sum_{p\isin{P}} (r_{a,p}-\bar{r}_a) (r_{b,p}-\bar{r}_b) }
{ \sqrt{ \sum_{p\isin{P}} (r_{a,p}-\bar{r}_a)^2}
  \sqrt{ \sum_{p\isin{P}} (r_{b,p}-\bar{r}_b)^2}
  } $$

Per correlazioni non lineare e variabili non gaussiani si utilizza ***Spearman***

#### Similarità coseno

Dati:
- a, b: sono due utenti
- r<sub>a,p</sub> e r<sub>b,p</sub>: rispettivamente rating dell'utente a e dell'utente b per il prodotto p
- P: set di prodotti per cui sia a che b hanno espresso un giudizio
- r<sub>a</sub>: vettore dei rating del'utente a dei prodotti P

Il risultato ha valori compresi tra 0, massima dissimilarità, e 1, massima similarità.  
La formulazione è la seguente:
$$ sim\_cosine(a,b) = 
\frac
{ \sum_{p\isin{P}} (r_{a,p} \cdot r_{b,p})}
{ \sqrt{\sum_{p\isin{P}}(r_{a,p})^2}
  \sqrt{\sum_{p\isin{P}}(r_{b,p})^2}
  } $$

#### Generare una raccomandazione

Dati:
- Sia n l'insieme degli utenti più simili all'utente a
- (r<sub>b,p</sub> - r<sub>b</sub> barrato): Per ogni utente b appartenente ad N, calcoliamo la differenza tra il giudizio su p di b e il giudizio medio di b sui prodotti valutati dall'utente a e b
- sim(a,b) è la similarità tra gli utenti a e b che serve a pesare il giudizio differenza di b usando come peso la similarità con a
- r<sub>a</sub> barrato è la media dei giudizi espressi da a

Conoscendo le formule per calcolare la similarità, per predire il prodotto p da raccomandare all'utente a si utilizza la seguente formula:
$$ pred(a,p)=\bar{r}_a + \frac {\sum_{b\isin{N}} sim(a,b)\cdot(r_{b,p}-\bar{r}_b)} {\sum_{b\isin{N}}sim(a,b)}$$

#### Migliorare le metriche di predizione

Non tutte le valutazioni degli utenti simili soo ugualmente utili. Una possibile soluzione è quella di dare maggior peso agli elementi sui quali esiste maggiore varianza sui relativi voti. Ha quindi molta importanza il numero di elementi _co-rated_, ad esempio prodotti apprezzati da entrambi gli utenti.  
Si può anche amplificare il peso degli utenti "molto simili", cioè con un valore di similarità vicino ad 1.

#### Approcci memory e model based

Il CF user-based è considerato _basato sulla memoria_ poiché la matrice di valutazione viene direttamente utilizzata per trovare gli utenti simili e per fare previsioni. Questo approccio può non scalare per la maggior parte degli scenari del mondo reale.  
Per questo è meglio utilizzare un approccio model-based, cioè basati su una fase di pre-elaborazione o _model-learning_:
- Il modello addestrato è usato per fare previsioni
- I modelli sono aggiornati e riaddestrati periodicamente
- La costruzione di modelli e l'aggiornamento può essere computazionalmente costosa

### Item-based CF

Un approccio model-based è quello _item-based CF_.
L'idea di base è quella di utilizzare la similarit tra prodotti (e non utenti) per fare prevedere quale voto darebbe l'utente _u_ al prodotto _p_.  
I parametri sono simili a quelli dell'approccio _user-based_. Per questo caso, è dimostrato che il calcolo della similarità tramite coseno funziona meglio.  
La dimensione dell'insieme degli item simili per fare la previsione è in genere fissa e minore dell'insieme completo dei simili.  
Una tecnica comune per predire la raccomandazione, con dati:
- u : utente
- p : prodotto
- r<sub>u,i</sub> : rate di u del prodotto i
- ratedItem(u) : prodotti valutati da u

è la seguente:
$$ pred(u,p) = \frac 
{\sum_{i\isin{ratedItem(u)}}sim(i,p)\cdot{R}_{u,i}}
{\sum_{i\isin{ratedItem(u)}}sim(i,p)}$$

L'approccio item-based non risolve il problema della scalabilità anche se le similarità tra prodotti dovrebbero essere più stabili di quelle tra utenti.  
Un altro problema è quello del _cold start_. Come si possono raccomandare nuovi elementi? Cosa si può consigliare ai nuovi utenti? Esistono vari approcci:
- Diretti
  - Chiedere/forzare gli utenti a valutare una serie di prodotti
  - Utilizzare un metodo content-based oppure on personalizzato, ossia basato sulla popolarità generale dei prodotti
  - Semplicemente non raccomandare nulla
- Utilizzare algoritmi specifici (non nearest neighbor)
  - Nell'approccio NN, il set dei simili sufficientemente simili potrebbe essere troppo piccolo per fare buone previsioni. Assumere quindi una "transitività" dei simili, cioè A simile a B, B a C, quindi A simile a C.
- Approccio a dati sparsi, _Recursive CF_
  - Se esiste un utente _u1_ molto simile all'utente considerato _u_, che però anch'esso non ha votato per un prodotto _p_, utilizzare CF per trovare il rating dell'utente _u1_ per _p_. Infine utilizzare _u1_ nel calcolo dei simili a _u_, invece che utilizzare utenti molto diversi a causa della sparsità dei dati

### Metodi _Graph-based_

_Spreading activation_ sfrutta la "transitività" presunta dei gusti del cliente e, quindi, aumenta la matrice con informazioni aggiuntive.  
Supponiamo di dover raccomandare un prodotto per _u_. Con un approccio CF standard, _u1_ sarà considerato un vicino di _u_ perché entrambi hanno acquistato _p1_ e _p2_. Il prodotto _p_ sarà raccomandato a _u_ considerando gli acquisti del vicino più prossimo (_u1_).  
Sfruttando più in profondo la transitività, si possono utilzzare associazioni indirette con percorsi più lunghi. Cioè utilizzando ad esempio una transitività a 5 passi piuttosto che a 3 come nell'esempio di prima.

### Altri metodi model-based

Negli ultimi anni sono state proposte numeroso tecniche:
- Tecniche statistiche di matrix factorization come _SVD_ 
- Regole associative, utilizzate a.e. per la market basket analysis
- Modelli probabilistici, modelli di clustering, reti bayesiane, Latent Semantic Analysis probabilistico
- Approcci ad apprendimento aumetomatico più complessi

I costi di preprocessing non sempre sono sostenibili e sono un tema di ricerca.

### SVD - Singular Value Decomposition

L'idea di base è quella di creare modelli più complessi offline per la produzione di previsioni più veloci online.  
Si utilizza SVD per la riduzione della dimensionalità delle matrici di rating. SVD:
- genera un nuovo spazio con nuovi assi dove colloca i dati della matrice
- i nuovi assi rappresentano le dimensioni lungo le quali i dati variano maggiormente, tuttavia non sono sempre facilmente interpretabili
- cattura i segnali rilevanti nei dati filtrando il rumore con un numero _k_ di dimensioni molto inferiore alle dimensioni originali (20 <= _k_ <= 100)

Le raccomandazioni utilizzando SVD sono fornite in tempo costante.  

SVD afferma che una matrice _M_ può essere fattorizzata nel prodotto di 3 matrici:
$$ M = U \times \Sigma \times V^\intercal $$
Dove _U_ contiene gli autovettori destri di _MM<sup>T</sup>_, similarità tra utenti, e _V_ gli autovettori sinistr di _M<sup>T</sup>M_ (gli autovettori sono ortonormali e costituiscono la base di un nuovo sistema di riferimento i cui assi sono le variabili latenti).  
I valori della matrice diagonale Σ sono gli autovalori, detti anche valori singolari, in ordine decrescente sulla diagonale. Ogni autovalore è la varianza dei dati sulla dimensione che rappresenta.  
La matrice di partenza si può approssimare utilizzando nel prodotto delle 3 matrici i primi _k_ valori singolari maggiori in Σ.  

Riassumendo, con la matrix factorization SVD:
- si individuano i fattori latenti, ossia le dimensioni del nuovo spazio
- si generano approssimazioni di matrici a basso rango k (numero di autovalori scelti)
- si ha la proiezione di oggetti e utenti nello stesso spazio n-dimensionale

La qualità della previsione dipende dalla giusta sceclta di k (20 <= k <= 100), ossia dal numero dei valori singolari. I parametri possono essere determinati e ottimizzati solo su esperimenti in un determinato dominio, perciò l'accuratezza delle raccomandazioni può anche diminuire rispetto all'impiego della matrice di rating originale con valori di k inadeguati.

### Recommendation con regole associative

Sono utilizzate comunemente per l'analisi impersonale dei comportamenti di acquisto. Le regole associative si ricavano dall'analisi delle frequenze delle combinazioni di acquisto dall'intero insieme di transazioni di acquisto.  
Le regole associative sono implicazioni statistiche del tipo x->y.  
Esistono due misure di accuratezza delle regole associative, utilizzate anche come cut-off per scegliere le regole migliori:
- **supporto**: il supporto di una regola è il numero di transazioni (di tutti gli utenti) dove x è acquistato insieme ad y, diviso il numero totale di transazioni
- **confidenza**: la confidenza ha lo stesso numeratore del supporto, mentre il denominatore è il numero delle transazioni che contengono x.

### Metodi probabilistici

L'idea di base è "Qual è la probabilità che un utente esprima un rating _r_ di un prodotto _p_ dati i rating dell'utente e di tutti gli altri utenti?".  
Sia Y la variabile aleatoria relativa al rating su _p_ ed X i rating dell'utente.  
Occorre calcolare P(Y, X) = P(Y|X) P(X)=P(X|Y)P(Y) --> Bayes  
Con l'assunzione che i rating X<sub>i</sub> sui prodotti _d_ sono considerati indipendenti tra loro
$$
P(Y|X)= \frac {P(X|Y)\times{P(Y)}} {P(X)} 
\\
P(Y|X) = \frac {\prod^d_{i=1}P(X_i|Y)\times{P(Y)}} {P(X)}
$$
dato che P(X) è invariante rispetto ad Y ed è sufficiente determinare quel Y che massimizza
$$ \argmax_y\prod^d_{i=1}P(X_i|Y)\cdot{P(Y)} $$

### Slope One Predictor

Il predittore _Slope One_ è semplice e si basa su un differenziale di popolarità tra gli elementi per gli utenti. Lo schema di base è quello di prendere la media di queste _differenze dei co-rating_ per fare la previsione.  
In generale bisogna trovare una funzione della forma _f(x) = x + b_.  
Se un utente ha votato diversi elementi, le previsioni possono essere combinate utilizzando una media ponderata. Una buona scelta per il peso è il numero di utenti che hanno valutato entrambi gli elementi.

### Riassunto metodi di CF

**Pro**: facile da comprendere e implementare, funziona bene in alcuni settori.  
**Contro**: richiede comunità di utenti, ha il problema della sparsità, non c'è nessuna integrazione di altre fonti di conoscenza.  
Non esiste un metodo CF migliore di un altro, le differenze fra i metodi sono spesso molto piccole (ca. 1%).  
Sicuramente il metodo più utilizzato per valutare un RS è il RMSE.

# Classificazione

Classificare significa individuare una funzione che massimizzi la separazione tra le classi.

## Classificazione lineare con iperpiani

Sono metodi che assumono l'esistenza di separazioni lineari dei dati. Consistono nell'individuazione di _iperpiani di separazione_ delle classi con _programmazione lineare_ oppure con soluzioni iterative, come a.e. _perceptron_, _newton_, _discesa del gradiente_.  
Un iperpiano _**w⋅x**+b=0_ partiziona lo spazio in 2 parti ed è definito da 2 parametri:
- _w_ vettore unitario perpendicolare all'iperpiano di separazione
- _b_ distanza ortogonale dell'iperpiano dall'origine, detta anche intercetta

In 2D un iperpiano è una retta.  
- Le istanze x<sup>(i)</sup> (vettori) che giacciono sull'iperpiano soddisfano l'equazione _**w⋅x**<sup>(i)</sup>+b=0_, perciò la loro distanza ortogonale dall'origine è _|b|_ ossia ***w⋅x***_(i)=-b_.   
La distanza ortogonale è uguale alla distanza dalla retta parallela che passa in (0,0) data da ***w⋅x***_=0_.
- Le istanze ***x***<sup>(+)</sup> tali che _**w⋅x**<sup>(+)</sup>+b>0_ sono della classe (+) al di sopra dell'iperpiano con distanza > -b.
- Le istanze ***x***<sup>(-)</sup> tali che _**w⋅x**<sup>(-)</sup>+b<0_ sono della classe (-) al di sotto dell'iperpiano.

### Alcune caratteristiche dei classificatori

Esistono molte possibili soluzioni con parametri distinti **w**, b. Alcuni metodi individuano un iperpiano di separazione non ottimale, in base ad un qualche criterio di ottimalità, altri individuano un iperpiano di separazione ottimale.  
Quali dati influenza la ricerca dell'iperpiano?
- Tutti i punti nei metodi
  - Perceptron, Regressione Logistica, Naive Bayes, Regressione lineare con soglia
- Solo i "punti difficili", a.e. vicini al decision boundary
  - Support Vector Machines (_SVM_)
  
Esistono poi problemi con dati non separabili linearmente, risolvibili con numerosi approcci:
- Soluzioni che trasformano lo spazio dei dati in modo che le classi diventino separabili linearmente
  - SVM, Reti neurali...
- Soluzioni intrinsecamente non lineari
  - kNN, Decision Tree, RandomForest, Gradient Boosting, XGboost, lightGBM, Catboost...
  
In generale, più variabili hanno i dati, più chance ci sono di separare le classi linearmente.

### Perceptron

Percpetron è il progenitore delle reti neurali. Converge solo se i dati sono separabili linearmente.  
Prendiamo come esempio la classificazione di un tumore in maligno/benigno.  
Variabili di input:
$$ x_1 = mean\_area \\ x_2 = mean\_concave\_points$$
Variabile da predire _discreta_:
$$ y=\begin{cases}
  -1 \ if \ benign\\
  +1 \ if \ malign
\end{cases} $$ 

La separazione lineare delle due classi è data da:
- Retta di separazione con _w<sub>1</sub>, w<sub>2</sub>, b_ da apprendere
  - $$ b +w_1x_1+w_2x_2 = 0 $$
- Le cellule benigne sono le coppie:
  - $$ x = (x_1,x_2) \ t.c. \ b+w_1x_1+w_2x_2 \lt 0 $$
- Mentre qulle maggiori o uguali a zero sono maligne, ossia si ha:
  - $$ y = \begin{cases}
            -1 \ if \ b+w\cdot{x}\lt0 \\
            +1 \ if \ b+w\cdot{x}\ge0
            \end{cases} $$

Come possiamo trovare l'iperpiano di separazione? Dato che ogni istanza è etichettata con -1 o +1, possiamo riscrivere in forma compatta _-y(b+w⋅x)<0_ che darà come risultato un valore negativo se correttamente classificata.  
Quindi possiamo scrivere _max(0, -y(b+w⋅x)<0) = 0_ che restituisce 0 se l'istanza è correttamente classificata.  
Di conseguenza possiamo minimizzare l'errore sulle _m_ istanze di training con:
$$ \underset{b,w}\text{minimize}\sum_{i=1}^m\max(0, -y_i\cdot{h_w(x_i))} \text{ dove } h_w(x_i)=b+w\cdot{x_i}$$
La funzione suddetta è continua e convessa ma _non derivabile_, perciò il metodo di discesa del gradiente è inapplicabile, inoltre ha un min fittizio per b=w=0.  

#### Logistic Loss

Sostituiamo la funzione di prima _max(0, s)_ con una funzione derivabile. Una sua approssimazione è nota come _softmax(0, s) = log(1+e<sup>s</sup>) che è convessa, continua e derivabile.  
Perciò minimizzando la somma rispetto a _b_ e _w_ della softmax su tutte le istanze, si minimizza l'errore. Questa formulazione è conosciuta come _logistic loss_ a cui si può aggiungere la regoralizzazione L2 (o L1) dei parametri _w_ con peso λ:
$$
  \text{Senza regolarizzazione} \\
  \underset{b,w}\text{minimize}\sum_{i=1}^m{\log{(1+e^{-y_i\cdot{h_w(x_i)}})}} \\ 
  \text{Con regolarizzazione L2} \\
  \underset{b,w}\text{minimize}\sum_{i=1}^m{\log{(1+e^{-y_i\cdot{h_w(x_i)}})} +\frac{\lambda}{2}||w||^2_2}
$$

## Regressione logistica o Sigmoide

La regressione logistica deriva dalla logistic loss con norma L2 in forma compatta (senza _b_). Può essere considerata la versione moderna del Perceptron.  
$$ \sigma(t) = \frac{1}{1+e^{-(b+w\cdot{t})}} \\
\text{Dominio e Codominio: } \R\rarr[0,1]$$
Il risultato è intepretabile come probabilità di appartenenza di ogni istanza _t_ ad una delle classi, approssima una funzione a gradino.

### Regressione logistica in due classi: multivariata lineare e non lineare

Sia
$$ x\isin{\R^n} \text{ un'istanza di } x=(x_1,...,x_n) \text{ in } n \text{ variabili}$$
La funzione di _classificazione lineare multivariata_ è
$$ \sigma(x_1,...,x_n)=\frac{1}{1+e^{-h_w(x_1,...,x_n)}} \\
\text{ dove } h_w(x_1,...,x_n)=w_1x_1+...+w_nx_n+b \text{ è l'iperpiano di separazione}$$

La _regressione logistica non lineare_ è, a.e. con grado 2:
$$ \sigma(x_1,...,x_n)=\frac{1}{1+e^{-h^2_w(x_1,...,x_n)}} \\
\text{ con } h^2_w(x_1,...,x_n)= \sum_{i=1}^n{w_{i,1}x_i^2}+2\sum_{i=1}^{n-1}{\sum_{j=i+1}^n{w_{i,j}x_ix_j}}$$
Sul numero di termini generati in n variabili con grado _g_ e non scalabilità, vale quanto già visto nella regressione non lineare. Il problema è stato però superato perché esiste la _Kernel Logistic Regression SVM_.

## Classificazione multiclasse e iperpiani
Esistono due metodi con _C_ classi:
- One-Versus-All
  - Si individuano _C_ iperpiani, uno per ogni classe da separare da tutte le altre, con lo stesso metodo di individuazione di un singolo iperpiano (è parallelizzabile)
    - $$ b_c+x^Tw_c=0, \ c=1,...,C$$
  - La regola di fusione è la seguente: ad ogni istanza _x_ si assegna la classe _y_ corrispondente al piano _j_ che massimizza
    - $$ y=\argmax b_j+x^Tw_j, \ j=1,...,C$$
- Multinomial
  - Individua congiuntamente _C_ iperpiani minimizzando la regola di fusione di cui sopra. Quindi, ogni istanza x<sub>p</sub> è classificata correttamente nella propria classe _c_ se
    - $$ \max_{j=1,...,C}(b_j+x^T_pw_j)-(b_c+x_p^Tw_c)=0$$
    - la parte sinistra è maggiore di zero solo con errori, perciò è la loss da minimizzare
  - Per derivarla sostituiamo max con softmax e minimizziamo la seguente funzione
    - $$ \text{Con  } soft(s_1,...,s_C)=\log{\sum_{j=1}^C}e^{s_j}\\
    \sum_{c=1}^C\sum_{p\isin\Omega_c}[\max_{j=1,...,C}(b_j+x^T_pw_j)-(b_c+x_p^Tw_c)] = -\sum_{c=1}^C\sum_{p\isin\Omega_c}\log(\frac{e^{b_c+x_p^Tw_c}}{\sum_{j=1}^Ce^{b_j+x_p^Tw_j}})
    $$

## Classificazione con classi sbilanciate

In molti problemi reali la suddivisione di istane tra classi è molto sbilanciata (a.e. transazioni con carte di credito, lecite >> illecite), di conseguenza si hanno molti più errori di classificazione sulla classe meno rappresentata, spesso con risultati inaccettabili.  
Una possibile soluzione è quella di aumentare il peso degli errori sulla classe con meno istanze.

## Individuazione dell'iperpiano

L'iperpiano individuato da Perceptron e Regressione logistica è uno dei tanti possibili iperpiani di separazione, finora non abbiamo mai stabilito un criterio per stabilire quale tra le soluzioni sia la migliore, al di là della minimizzazione del numero di errori sul training set. Tuttavia ricordiamo che la regolarizzazione vincola e riduce le possibili soluzioni.  
Quindi, come definiamo il concetto di migliore?  
Il numero di soluzioni diminuisce se cerchiamo una separazione lineare con il maggiore margine possibile tra le istanze delle due classi, possiamo quindi individuare l'iperpiano che massimizzi il margine (minore overfitting).  
L'iperpiano migliore è definito dai punti _difficili_ chiamati _support vectors_, cioè tutti quei punti vicini al _decision boundary_. Se mancassero i restanti punti (tutti i _non_ SV), l'iperpiano calcolato sarebbe il medesimo.  
A questo punto, il problema diventa risolubile come problema di ottimizzazione quadratica attraverso l'utilizzo delle _Support Vector Machines_.

## Valutazione di modelli di classificazione

I modelli di classificazione si valutano sul test/validation set calcolando una matrice 'Classe reale' X 'Classe Predetta'. 

### Classificazione e tasso di errore

Il tasso di errore calcolato sul training set è inevitabilmente ottimistico rispetto all'errore atteso su nuovi dati. Per questo, i dati nei problemi reali sono suddivisi in tre subset:
- Training set
- Validation set, usato per fare il tuning degli iperparametri
- Test set, utilizzato per simulare il tasso di errore atteso sui nuovi dati

Supponiamo, ora, che un classificatore abbia un tasso di successo sul test set, i.e. accuratezza, del 75%. Quanto è attendibile questa accuratezza sull'intera popolazione dei dati, compresi quelli nuovi ignoti?  
La risposta non sarà un singolo valore, piuttosto un intervallo: 75% +- ?.  
L'intervallo di accuratezza dipende dalle dimensioni del test set, ma quanto le dimensioni del test set influenza l'intervallo di accuratezza?  
In ogni progetto di data science con modelli predittivi, queste misure sono essenziali per valutare l'affidabilità del risultato.  

### Modellazione della classificazione come Processo di Bernoulli

La classificazione di N istanze è modellabile con un processo Bernolulliano di N eventi binari indipendenti: errore o successo.  
Per esempio, testa o croce nel lancio di una moneta: se con 100 lanci abbiamo 75 teste, qual è la probabilità _p_ di ottenere testa nel prossimo lancio? E dopo 1000 lanci?  
Siano _N_ gli esperimenti, _S_ i successi. Si ha _f_ = _S_/_N_ --> tasso di successo (accuratezza).  
_f_ ha distribuzione binomiale _Bin(N, p)_ con media _p_ e varianza _p(1-p)_/_N_, _p_ è la reale accuratezza che vogliamo stimare.  
Per _N_ grande (>30) la distribuzione di _f_ è approssimabile con la distribuzione normale standardizzata (_distribution z_). La confidenza è:
$$ Pr[-z\lt\frac{f-p}{\sqrt{p(1-p)/N}}\lt z] = c$$

### Confrontare l'accuratezza di due modelli

Dati due modelli _M1_ e _M2_, qual è il migliore?
- _M1_: testato su un dataset _D1_ di cardinalità _n1_ con errore _e1_
- _M2_: testato su un dataset _D2_ di cardinalità _n2_ con errore _e2_
Se _n1_ ed _n2_ sono abbastanza grandi (>30) il loro errore è approssimabile da una distribuzione normale con media μ e deviazione standard σ.
$$ e_1\approx N(\mu_1,\sigma_1) \\ e_2\approx N(\mu_2,\sigma_2) 
\\ \text{la varianza approssimata è: } \\ \hat\sigma^2_i=\frac{e_i(1-e_i)}{n_i}$$
Come valutare però se la differenza _d_ tra le accuratezze dei due modelli è statisticamente significativa?  

Sia _d_ = _e1 - e2_, d ~ N(d<sub>t</sub>,σ<sub>t</sub> dove d<sub>t</sub> è la reale differenza da stimare.  
La varianza si ottiene come segue:
$$ \sigma^2_t = \sigma_1^2 + \sigma_2^2 \approxeq \hat\sigma_1^2 + \hat\sigma_2^2 = \frac{e1(1-e1)}{n1}+\frac{e2(1-e2)}{n2}$$
Infine d<sub>t</sub> (con confidenza 1-α) è:
$$ d_t = d \plusmn Z_{\frac{\alpha}{2}}\hat\sigma_t $$

# Natural Language Processing

## Dati strutturati e destrutturati

I dati strutturati sono tutti quei dati provvisti di un modello, o schema, capace di descriverli e attribuirgli semantica. Ad esempio il rating di un utente per un prodotto visto nella recommendation.  
I dati destrutturati sono invece tutti quei dati sprovvisti di un modello in grado di spiegarne formalmente la semantica. Ad esempio le recensioni di un prodotto scritte in linguaggio naturale.  

Il dato destrutturato di maggior interessa è il testo in linguaggio naturale (a.e. italiano). Il testo segue regole linguistiche come sintassi, lessico, grammatica, ma in generale è_ambinguo_ e dipendente dal contesto, senza considerare la possibile presenza di errori (refusi, punteggiatura, verbi sbagliati...) e l'utilizzo costante di elementi non previsti dalla lingua (emoticon, hashtag, abbreviazioni...). I dati testuali destrutturati sono ampiamente diffusi nel mondo del web e la conoscenza ricavabile da essi può avere grande valore.

## Pre-processing di un testo

Del testo è inizialmente disponibile come una sequenza di caratteri in forma non strutturata. Un primo passaggio per l'analisi consiste nel convertirlo in una forma più facilmente interpretabile dal calcolatore.

### Segmentazione (_tokenization_)

La prima operazione compiuta in genere sul testo è la sua _segmentazione_ in elementi sinstattici di base (_token_). L'operazione più comune consiste nello scomporre il testo nelle sue singole parole (_word tokenization_). Testi non brevi possono anche essere dapprima divisi in frasi (_sentence tokenization_).  

#### Word tokenization
Le parole in una frase si possono separare semplicemente considerando spazi e segni di punteggiatura, esistono però casi particolari e ambigui ("isn't", parole separate da un trattino...). Per una segmentazione accurata è quindi utile integrare regole e modelli specifici della lingua analizzata.  

#### Part of Speech _POS_

Le _part of speech_ sono le classi crammaticaali a cui ciascun elemento di una frase può appartenere (nome, verbo, aggettivo...). Esistono diverse tassonomie di POS, da utlizzare a seconda delle esigenze. La più nota è quella usata dal Penn Treebank.  

Il _POS tagging_ consiste nell'etichettare ciascuna parola estratta dalla segmentazione del testo col suo POS nella frase. Questa operazione deve tenere conto del contesto di ogni parola, in quanto una stessa parola può avere diversi POS. Il POS tagging può essere utile o meno a seconda dei passaggi successivi: è utile ad esempio se si vogliono filtrare parole di tipo specifico o se elaborazioni successive dipendo dai POS mentre non è necessario ad esempio se si vogliono etrarre le parole chiave del documento indipendentemente dalla loro posizione nella fase.  

La seguenza delle parole estratte va spesso ripulita prima di essere utilizzata nelle analisi successive poiché alcuni elementi estratti possono non essere necessari (punteggiatura, html...) e altri elemento possono essere "normalizzati", cioè si possono eliminare le differenze di forma ma non di sostanza (coniugazioni diverse dello stesso verbo...). Per questo, dopo l'estrazione dei token, è spesso prevista una serie di passaggi per modificare la sequenza.  

#### Pulizia

##### Casefolding

Il casefolding consiste nel convertire tutte le parole completamente in minuscolo, è un metodo basilare per uniformare token.

##### Rimozione stopword

Alcune parole di un documento, indipendentemente dal loro ordine, sono indicative dell'argomento trattato. Altre parole, a.e. articoli, preposizioni e congiuzioni, non danno indiciazioni sull'argomento. Queste parole, dette _stopword_, sono spesso rimosse dalla pappresenzatazione di ciascun documento. Ciò si basa su una lista predefinita di stopword, chiamata _stoplist_.

##### Lemmatizzazione

La lemmatizzazione consiste nel sostituire ciascuna parola col suo _lemma_, ovvero la sua forma base presente nel dizionario.

##### Stemming

Diversamente dalla lemmatizzazione, lo _stemming_ ricava da ogni parola la sua radice morfologica. La radice può non essere una parola di senso compiuto mentre il lemma lo è.  
Lo stemming si usa come alternativa più efficiente alla lemmatizzazione per unificare termini simili. Ovviamente lo stemming è più approssimativo ed il suo utilizzo può comportare una perdita di precisione.

## Full Text Search

Le interrogazioni viste finora su database sono eseguite su dati strutturati con ottimizzazione dell'efficienza. L'information Retrieval si occupa della ricerca di testo in documenti, ossia in testo non strutturato, ciò richiede l'estrazione e indicizzazione die termini in ogni documento. L'indicizzazione è preceduta dalle teniche di pre-processing suddette.  
Se la query contiene più termini, occorre specificare come si combinano. Devono compararire tutti? Devono essere vicini?  
Si possono volere individuare anche termini simili a quelli scritti esplicitamente.  
Per soddisfare al meglio la query dell'utente i documenti individuati devono essere elencati a partire dal più rilevante e la ricerca deve essere il più efficiente possibile.

### Compromesso tra esattezza e completezza

Data una o più query di test (con documenti rilevanti noti), è possibile misurare la bontà del sistema in termini di:
- ***precision***: percentuale di documenti rilevanti tra quelli restituiti
- ***recall***: percentuale di documenti rilevanti restituiti tra tutti quelli rilevanti nella collezione

Tendenzialmente, cambiando i parametri di un sistema _FTS_ per migliorare uno dei due aspetti, l'altro peggiora.

### Modelli

La disciplina dell'Information Retrieval ha elaborato diversi modelli per rappresentare e recuperare documenti. Data un'interrogazione, il modello determina quali sono i documenti rilevanti e/o quanto ciascuno di essi lo è.  
I modelli proposti possono essere classificati in base alle teorie matematiche su cui sono definito o in base alla considerazione dell'interdipendenza dei termini.

#### Modello booleano standard

Nel modello booleano, i documenti sono semplicemente divisi tra rilevanti e non ad un'interrogazione, senza un ordine preciso.  
Ad un generico termine di ricerca viene associato un insieme di documenti rilevanti, quelli che contengono il termine.  
I termini posso essere combinati con operatori logici, a cui corrispondo diverse operazioni sugli insiemi:
- A _AND_ B (congiunzione logica): _intersezione_ tra l'insieme di documenti rilevanti per A quelli rilevanti per B
- A _OR_ B (disgiunzione logica): _unione_ tra gli insiemi di documenti
  
Il pro di questo modello è che è concettualmente semplice e facile da implementare. Di contro ha che restituisce i documenti senza un ordine significativo e spesso in quantità troppo alta o troppo bassa.  

#### Estrarre la semantica di un test

Una comprensione completa ed esatta del significato di un testo è molto complessa per un calcolatore, è opportuno utilizzare tecniche che simulino il ragionamento umano, quali ad esempio il _deep learning_. D'altra parte però la comprensione esatta è spesso non necessaria.  
Per capire il significato di un testo, è necessario conoscere le parole esistenti e come queste siano in relazione tra loro. Gli algoritmi per l'analisi testuale (soprattutto semantica) usano spesso delle basi di conoscenza esterne del linguaggio.

#### WordNet

_WordNet_ è tra i più noti database lessicali della lingua inglese.  
Le principali relazioni semantiche in _WordNet_ sono:
- Iponimia: "essere un (tipo specifico di)", _is-a_
- Meronimia: "essere parte/membro/sostanza di"
- Implicazione: un'azione ne comporta un'altra
- Antonimia: "essere opposto di" (relazione lessicale)

#### Word Sense Disambiguation

La _Word Sense Disambiguation_ è la procedura che associa ad ogni parola in un testo il suo significato esatto. La _WSD_ richiede una base di conoscenza che associ ad ogni parola i significati possibili.  
La disambiguazione avviene in genere in base al contesto di ciascuna parola, ovvero le parole vicine ad essa. Ad esempio, l'algoritmo di Lest considera il significato la cui definizione ha il maggior numero di parole presenti vicino al termine nel testo.  

#### Named Entity Recognition

Una _named entity_ è una specifica entità a cui ci si può riferire per nome in un testo. La _named entity recognition_ consiste nell'individuare le named entity in un testo e classificarle per tipo.

#### Rappresentare il contenuto di un testo

Nell'analisi di testi serve spesso rappresentare il contenuto generale di un documento in modo strutturato e compatto. Le parole contenute in un documento sono in genere indicative del suo contenuto. D'altra parte, non sono strettamente necessarie informazioni quali l'ordine delle parole o la loro _POS_.

##### Bag of Words

Una _Bag of Words_ (_BOW_) è una rappresentazione in forma di multiset delle parole contenute in un'unità di testo (un documento). Si tratta di un elenco delle parole distinte presenti nel testo, con associata a ciascuna il numero di occorrenze in esso.  
Normalizzando le parole in fase di pre-processing, si riduce il numero di parole distinte, semplificando i passi successivi.  

###### _n_-gram

Gli _n_-gram sono sequenze di _n_ parole presenti in un testo, se _n_=2 si parla di _bigram_, se _n_=3 si parla di _trigram_.  
Possono rendere più completo il _BOW_ di un testo, includendo termini composti di più parole ("New York", "Bag of Words"...), sono però inclusi anche molti _n_-gram non significativi e il numero totale è molto alto.  

###### Rappresentazione vettoriale

Si consideri di definire un insieme _D_ (dizionario) di termini distinti che si potrebbero trovare in un documento di testo. Il _BOW_ di ciascun documento può essere rappresentato come un vettore di |_D_| elementi in cui l'_i_-esimo valore è il numero di occorrenze dell'_i_-esimo termine o in alternativa si può considerare un vettore binario con 1 per le parole presenti e 0 per le assenti.

#### Vector Space Model

Il _Vector Space Model_ prevede di rappresentare un insieme di _N_ documenti come vettori in un medesimo spazio. Lo spazio ha un numero di dimensioni _D_ pari al numero di parole considerate nel dizionario condiviso. L'insieme di documenti si può rappresentare in una matrice termini-documenti di dimensioni _DxN_ in cui l'elementi _i,j_ indica quante volte la parola _i_ appare nel documento _j_.

#### Term weighting

Ogni documento contiene termini più o meno significativi per caratterizzare il suo contenuto. Il numero di occorrenze costituisce un'indicazione immediata del "peso" di un termine all'interno di un documento, esistono però diversi schemi di _term weighting_ per ottenere pesi più accurati sulla base di altre informazioni.

#### TF-IDF e Vector Space Model

Nel _VSM_ occorre determinare come assegnare i valori (pesi) a ciascunt termine in ciascun vettore.  
_tf_ (term frequency) è il fattore locale, che pesa la rilevanza di ciascun termine in ciascuno dei singoli documenti, è cioè il numero di apparizioni del termine nel documento.  
_idf_ (inverse document frequency) è il fattore globale, che pesa l'importanza di ciascun termine nell'intera collezione, è quindi più alto per termini che compaiono in meno documenti, in quanto più utili a distinguere questi documenti dagli altri.  
Il _tf-idf_ per un termine _t_ e un documento _d_ è il prodotto dei logaritmi (log perché l'importanza dei termini non cresce linearmente con la loro frequenza) di _tf_ e _idf_.
$$ tf.idf(t,d)=\log(f(t,d))\cdot log(\frac{|D|}{|d\isin{D}:t\isin{d}})$$

#### Normalizzazione dei vettori

Per garantire pari peso a documenti di lunghezze diversa, si usa normalizzare i vettori ottenuti in seguito al term weighting
$$ w(t,d)_{norm} = \frac{w(t,d)}{\sqrt{\sum_{t\isin{D}}w(t,d)^2}} $$
La normalizzazione mantiene la direzione del vettore, che indica la frequenza relativa tra le parole e quindi l'argomento.

#### Similarità coseno

Rappresentando i documenti come vettori nel medesimo spazio, possiamo misurarne la similarità con diverse metriche. Per confrontare coppie di documenti si usa spesso la similarità coseno, pari al cosegno dell'angolo tra i vettori
$$ cos(a,b)=\frac{a\cdot b}{||a|| \cdot ||b||} = 
\frac{\sum_{i=1}^na_i\cdot b_i}{\sqrt{\sum_{i=1}^na_i^2}\cdot \sqrt{\sum_{i=1}^nb_i^2}}$$
Per vettori con tutti valori positivi, la similarità coseno è sempre inclusa tra 0 (angolo 90°, i documento non hanno parole in comune) e 1 (angolo 0°, i vettori hanno la stessa direzione).

#### Vector Space Model vs Modello booleano

In risposta ad una query, col _VSM_ si ottiene un ranking effettivo dei documenti, dal più rilevante a scendere. Non è però possibile usare espressioni booleane AND e OR! Per questo esistono modelli estesi che combinano _VSM_ e logica booleana: il modello booleano esteso prima recupera i documenti che soddisfano la proposizione booleana, poi il risultato è ordinato calcolando i rank dei documenti. Quindi il modello esteso aumenta il potere espressivo del _VSM_ con proposizione booleane.  

#### Modelli con dipendenze tra termini

Nei modelli visti finora, ciascun termine è considerato a se stante e indipendente da tutti gli altri. In altri modelli sono considerate esplicitamente dipendenze reciproche tra termini.  
La _Latent Semantic Analysis_ traspone documenti e query in un nuovo spazio dove le dimensioni corrispondono idealmente a concetti semantici, ciascuno dato da un autovettore nello spazio originale.  

#### Ricerca di un termine 

Per ottenere la lista di documenti rilevanti ad una query, bisogna individuare in quali documenti compaiono i termini ricercati. La soluzione più facile, quando si vuole ricercare un termine, è una ricerca sequenziale su tutti i documenti, verificando la rilevanza di ciascuno alla query. Questo comporta necessariamente tempi lunghi.

##### Indicizzazione

Per ricerche più efficienti si utilizzano degli indici.  
Una struttura tipica è l'indice inverso (_inverted index_) che associa ad ogni termine la lista di documenti in cui compare. Dato un termine, l'accessio avviene in un tempo minimo e costante _O(1)_ alla lista di documenti.  
L'indice si costruisce e mantiene aggiornato associando un documento a ciascun termine che appare in esso

## Machine Learning e classificazione

Le tecniche di _machine learning_ permettono di apprendere conoscenza dai dati per effettuare predizioni su di essi_.  
Un problema comune consiste nel _classificare_ i dati, ovvero dividerli tra due o più gruppi (classi) secondo criteri specifici.   
Per automatizzare questa operazione si possono usare algoritmi di apprendimento supervisionato: l'algoritmo analizza un insieme di dati (training set) pre-etichettati con i rispettivi gruppi di appartenenza ed estrare un modello di conoscenza; il modello è usato come classificatore per predire le classi di appartenenza di dati simili.

### Algoritmi di apprendimento supervisionato

Esiste una grande varietà di algoritmi di apprendimento, che generano diversi tipi di modelli di conoscenza:
- modelli probabilistici (a.e. _naive Bayes_): calcola la classe di appartenenza più probabile in base alle caratteristiche dell'oggetto
- k-nearest neighbor: verifica quale sia la classe più ricorrente tra i k oggetti del training set con caratteristiche più simili a quello dato
- e altri come alberi decisionali, SVM, reti neurali...

Necessario è definire le caratteristiche (_feature_) degli oggetti, tramite un'adeguata rappresentazione strutturata. Una soluzione tipica è rappresentare ciascun oggetto come un vettore in un unico spazio multidimensionale.

## Esempio - Customer satisfaction da dati testuali

In un'attività di e-commerce, è importante sapere il grado di soddisfazione dei clienti in relazione ai singoli prodotti. A questo fine, le loro recensioni sono un dato importante.  
Leggendo le recensioni, si possono spesso individuare parole specifiche che indicano la soddisfazione o meno dell'utente.  
Sarebbe utile, avendo delle recensioni, estrarre in automatico conoscenza sulle parole usate nel contesto specifico; usando recensioni pre-etichettate come positive e negative, si potrebbero cercare le parole chiave tra quelle ricorrenti.  

Col _machine learning_ è possibile estrarre un classificatore da recensioni pre-etichettate come positive e negative. Il modello ottenuto potrà essere usato per stimare la polarità di altre recensioni dello stesso contesto.  
Per trattare i testi, è necessario rappresentarli come vettori: possiamo usare il Vector Space Model.  
La presenza di ciascun termine costituisce una feature dei testi. Per confrontare due testi tra loro si può usare la similarità coseno.  

Un problema importate è reperire tutte le informazioni rilevanti per ogni prodotto in modo sistematico. Molti servizi online danno l'accesso via API alle informazioni, in questo modo altri servizi possono farne uso in modo automatico.  
Ad esempio Twitter fornisce un API per l'accesso ai tweet pubblicati: da una ricerca sugli ultimi tweet pubblicati, otteniamo informazioni in tempo reale su qualsiasi prodotto, da cui possiamo misurare il grado di soddisfazione verso di esso.  
