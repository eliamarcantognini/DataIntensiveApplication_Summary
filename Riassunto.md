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






αβλθ⋅Σ