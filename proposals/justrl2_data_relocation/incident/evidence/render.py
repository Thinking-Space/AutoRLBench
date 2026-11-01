"""Render measured prefix/continuation curves only after fresh acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def events(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene-metadata',type=Path,required=True)
    parser.add_argument('--resume',type=Path,required=True)
    parser.add_argument('--proof',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    proof=json.loads(args.proof.read_text())
    if 'metrics' in proof: proof=proof['metrics']
    assert proof['score'] is True and proof['fresh'] is True
    assert proof['checkpoint_payloads_rehashed']=={'actor':True,'critic':True}
    scene=json.loads((args.scene_metadata/'SCENE.json').read_text())
    prefix=args.scene_metadata/'run/events.jsonl'
    assert sha(prefix)==scene['files']['run/events.jsonl']['sha256']
    before=[e for e in events(prefix) if e['kind']=='rollout_end']
    after=[e for e in events(args.resume/'run/events.jsonl') if e['kind']=='rollout_end']
    assert [e['step'] for e in before]==[0,1,2,3] and [e['step'] for e in after]==[4,5,6,7]
    data=[{'step':e['step'],'phase':'prefix' if e['step']<4 else 'reference_continuation','metrics':e['metrics']} for e in before+after]
    records=[e for p in sorted((args.resume/'run/optimizer_evidence').glob('*.jsonl')) for e in events(p)]
    counters={}
    for role in ('actor','critic'):
        rows=[r for r in records if r['role']==role]
        loaded=[r for r in rows if r['event']=='restore_loaded'];saved=[r for r in rows if r['event']=='restore_saved']
        assert len(loaded)==len(saved)==1 and loaded[0]['payload_exact'] and saved[0]['payload_exact']
        observed=[(3,loaded[0]['snapshot']['optimizer_steps'][0])]
        observed += [(r['rollout_id'],r['optimizer_steps'][0]) for r in rows if r['event']=='restore_optimizer_step']
        assert [s for s,c in observed]==[3,4,5,6,7] and [c for s,c in observed]==[4,5,6,7,8]
        counters[role]=observed
    args.out.mkdir(parents=True,exist_ok=True)
    evidence={'scope':'Actual prefix and its accepted reference continuation; no LLM agent evaluation',
              'scene_sha256':sha(args.scene_metadata/'SCENE.json'),'proof_sha256':sha(args.proof),
              'prefix_events_sha256':sha(prefix),'resume_events_sha256':sha(args.resume/'run/events.jsonl'),
              'steps':data,'optimizer_counters':counters,'evaluation':proof['evaluation']}
    (args.out/'curves.json').write_text(json.dumps(evidence,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    metrics=[('data/survivor_reward_mean','Observed training reward'),
             ('data/admitted_count','Admitted generated samples'),
             ('data/admitted_hard_fraction','Observed hard-sample fraction')]
    for ax,(key,title) in zip(axes.flat,metrics):
        for group,color,label in [(data[:4],'#6b7280','prefix'),(data[4:],'#1671b9','reference continuation')]:
            ax.plot([r['step'] for r in group],[r['metrics'][key] for r in group],'o-',color=color,label=label)
        ax.axvline(3.5,color='#bf7a23',linestyle='--',linewidth=1)
        ax.set(title=title,xlabel='Training step',xticks=range(8));ax.grid(alpha=.18)
        if key.endswith(('mean','fraction')):ax.set_ylim(-.05,1.05)
        else:ax.set_ylim(0,18);ax.set_yticks([0,4,8,12,16])
    ax=axes[1,1]
    for role,style,color in [('actor','o-','#1671b9'),('critic','x--','#b34c4c')]:
        ax.plot([r[0] for r in counters[role]],[r[1] for r in counters[role]],style,color=color,label=role)
    ax.set(title='Observed restored Adam counters',xlabel='Training step',xticks=range(3,8),yticks=range(4,9));ax.grid(alpha=.18);ax.legend()
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Preserved step-3 checkpoint → reference continuation\nDashed boundary: files relocated before the repair',fontsize=13)
    fig.savefig(args.out/'continuation-curves.png',dpi=170)
    fig.savefig(args.out/'continuation-curves.svg')
    plt.close(fig)
    print(json.dumps({'curve_steps':8,'resumed_updates_per_role':4,'output':str(args.out)}))


if __name__=='__main__': main()
