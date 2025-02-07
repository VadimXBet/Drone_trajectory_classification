import os
import numpy as np
import motmetrics as mm

def motMetricsEnhancedCalculator(gtSources, tSources):
  acc = mm.MOTAccumulator(auto_id=True)
  for gtSource, tSource in zip(gtSources, tSources):

    gt = np.loadtxt(gtSource, delimiter=',')
    t = np.loadtxt(tSource, delimiter=',')

    for frame in range(int(gt[:,0].max())):
    #   frame += 1 # detection and frame numbers begin at 1
      gt_dets = gt[gt[:, 0]==frame, 1:6]
      t_dets = t[t[:, 0]==frame, 1:6]

      if (gt_dets.size  == 0) and (t_dets.size  == 0):
        continue

      C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], max_iou=0.9)
      acc.update(gt_dets[:,0].astype('int').tolist(), t_dets[:,0].astype('int').tolist(), C)

  mh = mm.metrics.create()
  summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                     'recall', 'precision', 'num_objects', \
                                     'mostly_tracked', 'partially_tracked', \
                                     'mostly_lost', 'num_false_positives', \
                                     'num_misses', 'num_switches', \
                                     'num_fragmentations', 'mota', 'motp', 'num_matches'], name='acc')

  strsummary = mm.io.render_summary(summary,
      formatters={'mota' : '{:.2%}'.format},
      namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
               'precision': 'Prcn', 'num_objects': 'GT', \
               'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
               'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
               'num_misses': 'FN', 'num_switches' : 'IDsw', \
               'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP', 'num_matches' : 'NM'})
  print(strsummary)

if __name__ == "__main__":
    gt_path = 'test_videos/gt'
    gtSources = [os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path)) if gt_file.endswith(".txt")]

    t_path = ''
    tSource = [os.path.join(t_path, gt_file) for gt_file in sorted(os.listdir(t_path)) if gt_file.endswith(".txt")]
    motMetricsEnhancedCalculator(gtSources, tSource)
