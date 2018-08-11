import os
import time
from pracmln import query
from pracmln import MLN, Database, query

def main(arg='.'):
    pth = os.path.join(arg, 'wts.pybpll.smoking-train-smoking.mln')
    mln = MLN(mlnfile=pth, grammar='StandardGrammar')
    pth = os.path.join(arg, 'smoking-test-tiny.db')
    db = Database(mln, dbfile=pth)
    start = time.time()
    query(method='EnumerationAsk', mln=mln, db=db, verbose=False, multicore=False).run()
    t2 = time.time()-start
    print('exact inference test: {}'.format(t2))

if __name__ == '__main__':
    main()
    
