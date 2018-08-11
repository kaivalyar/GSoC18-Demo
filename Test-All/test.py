import os
from pracmln import MLN, Database
from pracmln import query, learn
from pracmln.mlnlearn import EVIDENCE_PREDS
import time

def test_inference_smokers(arg='.'):
    pth = os.path.join(arg, 'wts.pybpll.smoking-train-smoking.mln')
    mln = MLN(mlnfile=pth, grammar='StandardGrammar')
    pth = os.path.join(arg, 'smoking-test-smaller.db')
    db = Database(mln, dbfile=pth)
    for method in ('EnumerationAsk',
                   'MC-SAT',
                   'WCSPInference',
                   'GibbsSampler'
                   ):
        for multicore in (False, True):
            print('=== INFERENCE TEST:', method, '===')
            query(queries='Cancer,Smokes,Friends',
                  method=method,
                  mln=mln,
                  db=db,
                  verbose=False,
                  multicore=multicore).run()


def test_learning_smokers(arg='.'):
    pth = os.path.join(arg, 'smoking.mln')
    mln = MLN(mlnfile=pth, grammar='StandardGrammar')
    pth = os.path.join(arg, 'smoking-train.db')
    db = Database(mln, dbfile=pth)
    for method in ('BPLL', 'BPLL_CG', 'CLL'):
        for multicore in (True, False):
            print('=== LEARNING TEST:', method, '===')
            learn(method=method,
                  mln=mln,
                  db=db,
                  verbose=False,
                  multicore=multicore).run()

def runall():
    start = time.time()
    test_inference_smokers()
    test_learning_smokers()
    print('all test finished after', time.time() - start, 'secs')

def main():
    runall()

if __name__ == '__main__':
    main()
