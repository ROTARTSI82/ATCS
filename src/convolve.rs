// use crate::config::{FuncT, NumT};
//
// pub struct ConvolvePoolLayer
// {
//    pub kernel: Box<[NumT]>,
//    pub biases: Box<[NumT]>,
//
//    pub kern_size: i32,
//    pub stride: i32,
//
//    pub inp_width: i32,
//    pub inp_height: i32,
//    pub inp_chan: i32,
//
//    pub outp_chan: i32,
//
//    pub pool_size: i32,
// }
//
// impl ConvolvePoolLayer
// {
//    pub fn feed_forward(&self, inp: &[NumT], out: &mut [NumT], act: FuncT)
//    {
//       let out_w = self.inp_width/self.stride;
//       let out_h = self.inp_height/self.stride;
//       for w in 0..out_w
//       {
//          for h in 0..out_h
//          {
//             for c in 0..self.outp_chan
//             {
//                out[h * out_w * self.outp_chan + w * self.outp_chan + c] = 0 as NumT;
//                for x in 0..self.kern_size
//                {
//                   for y in 0..self.kern_size
//                   {
//
//                   }
//                }
//             }
//          }
//       }
//    }
// }
