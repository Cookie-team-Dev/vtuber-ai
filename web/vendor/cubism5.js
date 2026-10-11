/*!
 * Cubism 5 renderer for PixiJS.
 *
 * Uses the official Live2D Cubism Core (vendor/live2dcubismcore.min.js) for
 * parameter and drawable math and PixiJS for rendering. Enough of the
 * Cubism Web Framework is reimplemented here (motions, expressions, physics,
 * eye blinks, breath) to drive a model without the TypeScript framework.
 */
(function (global) {
  "use strict";

  var CORE = global.Live2DCubismCore;

  // ---------------------------------------------------------------- helpers

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  // Cubic bezier in value space (restricted beziers: control x at 1/3, 2/3).
  function bezierValue(v0, v1, v2, v3, t) {
    var a = lerp(v0, v1, t);
    var b = lerp(v1, v2, t);
    var c = lerp(v2, v3, t);
    var d = lerp(a, b, t);
    var e = lerp(b, c, t);
    return lerp(d, e, t);
  }

  function smoothstep(t) {
    t = clamp(t, 0, 1);
    return t * t * (3 - 2 * t);
  }

  // ------------------------------------------------------- motion sampling

  // motion3.json "Segments" is a flat array:
  //   [t0, v0, <segment>...]
  // each segment is a type code followed by its data:
  //   0 linear:  [t, v]
  //   1 bezier:  [x1, y1, x2, y2, t, v]   (x control points are ignored for
  //                                         restricted beziers, the default)
  //   2 stepped: [t, v]
  //   3 inverse stepped: [t, v]
  function sampleCurve(segments, time) {
    var n = segments.length;
    if (n < 2) return 0;
    if (time <= segments[0]) return segments[1];
    var p = 2;
    var pt = segments[0];
    var pv = segments[1];
    while (p < n) {
      var type = segments[p];
      if (type === 1) {
        // bezier: [x1, y1, x2, y2, t, v] — the x control points are
        // ignored for restricted beziers (the standard, AreBeziersRestricted)
        var cy1 = segments[p + 2];
        var cy2 = segments[p + 4];
        var t1 = segments[p + 5];
        var v1 = segments[p + 6];
        if (time <= t1) {
          var t = (time - pt) / (t1 - pt || 1e-6);
          return bezierValue(pv, cy1, cy2, v1, clamp(t, 0, 1));
        }
        pt = t1;
        pv = v1;
        p += 7;
      } else if (type === 0 || type === 2 || type === 3) {
        // linear / stepped / inverse stepped: [t, v]
        var t1b = segments[p + 1];
        var v1b = segments[p + 2];
        if (time <= t1b) {
          if (type === 0) {
            var k = (time - pt) / (t1b - pt || 1e-6);
            return lerp(pv, v1b, clamp(k, 0, 1));
          }
          if (type === 2) return pv; // stepped: value at segment start
          return v1b; // inverse stepped: value at segment end
        }
        pt = t1b;
        pv = v1b;
        p += 3;
      } else {
        break; // unknown segment type
      }
    }
    return pv;
  }

  // ------------------------------------------------------------ motion data

  function MotionEntry(data) {
    this.data = data;
    var meta = data.Meta || {};
    this.duration = meta.Duration || 0;
    this.fps = meta.Fps || 30;
    this.loop = meta.Loop !== false;
    this.curves = [];
    var curves = data.Curves || [];
    for (var i = 0; i < curves.length; i++) {
      var c = curves[i];
      this.curves.push({
        target: c.Target || "Parameter",
        id: c.Id || "",
        segments: c.Segments || [],
        fadeIn: c.FadeInTime != null ? c.FadeInTime : -1,
        fadeOut: c.FadeOutTime != null ? c.FadeOutTime : -1,
        paramIndex: -1,
        partIndex: -1,
      });
    }
  }

  // ---------------------------------------------------------- motion player

  var MotionPriority = { NORMAL: 0, FORCE: 1, IDLE: 2 };

  function MotionManager(model) {
    this._model = model;
    this._current = null;
    this._time = 0;
    this._priority = -1;
    this._fadeIn = 0.3;
    this._fadeOut = 0.3;
    this._cache = new Map(); // file -> Promise<MotionEntry>
  }

  MotionManager.prototype.load = function (file) {
    if (this._cache.has(file)) return this._cache.get(file);
    var promise = fetch(file)
      .then(function (res) {
        if (!res.ok) throw new Error("motion " + file);
        return res.json();
      })
      .then(function (json) {
        return new MotionEntry(json);
      });
    this._cache.set(file, promise);
    return promise;
  };

  MotionManager.prototype.start = function (entry, priority) {
    if (priority < this._priority && this._current) return false;
    this._current = entry;
    this._time = 0;
    this._priority = priority;
    return true;
  };

  MotionManager.prototype.stop = function () {
    this._current = null;
    this._priority = -1;
  };

  // Returns true while a motion is still running.
  MotionManager.prototype.update = function (dt) {
    var motion = this._current;
    if (!motion) return false;
    this._time += dt;
    var d = motion.duration;
    if (d <= 0) return false;
    var t = this._time;
    var done = false;
    if (t >= d) {
      if (motion.loop) {
        t = t % d;
      } else {
        t = d;
        done = true;
      }
    }
    var model = this._model;
    var values = model.coreModel.parameters.values;
    var parts = model.coreModel.parts.opacities;
    for (var i = 0; i < motion.curves.length; i++) {
      var curve = motion.curves[i];
      var value = sampleCurve(curve.segments, t);
      // per-curve fade, falling back to the motion-wide fade
      var fi = curve.fadeIn >= 0 ? curve.fadeIn : this._fadeIn;
      var fo = curve.fadeOut >= 0 ? curve.fadeOut : this._fadeOut;
      var w =
        smoothstep(this._time / fi) *
        smoothstep(Math.max(0, d - this._time) / fo);
      if (w <= 0) continue;
      if (curve.target === "PartOpacity") {
        if (curve.partIndex < 0) {
          curve.partIndex = model.partIndex(curve.id);
        }
        if (curve.partIndex >= 0) {
          parts[curve.partIndex] =
            parts[curve.partIndex] * (1 - w) + value * w;
        }
      } else {
        if (curve.paramIndex < 0) {
          curve.paramIndex = model.paramIndex(curve.id);
        }
        if (curve.paramIndex >= 0) {
          var idx = curve.paramIndex;
          values[idx] = values[idx] * (1 - w) + value * w;
        }
      }
    }
    if (done) {
      this._current = null;
      this._priority = -1;
      return false;
    }
    return true;
  };

  // -------------------------------------------------------- expression data

  function ExpressionEntry(data) {
    this.params = [];
    var list = data.Parameters || [];
    for (var i = 0; i < list.length; i++) {
      this.params.push({
        id: list[i].Id || "",
        value: list[i].Value || 0,
        blend: list[i].Blend || "Overwrite",
        paramIndex: -1,
      });
    }
  }

  ExpressionEntry.prototype.apply = function (model) {
    var values = model.coreModel.parameters.values;
    var mins = model.coreModel.parameters.minimumValues;
    var maxs = model.coreModel.parameters.maximumValues;
    for (var i = 0; i < this.params.length; i++) {
      var p = this.params[i];
      if (p.paramIndex < 0) p.paramIndex = model.paramIndex(p.id);
      var idx = p.paramIndex;
      if (idx < 0) continue;
      if (p.blend === "Add") {
        values[idx] = clamp(values[idx] + p.value, mins[idx], maxs[idx]);
      } else if (p.blend === "Multiply") {
        values[idx] = clamp(values[idx] * p.value, mins[idx], maxs[idx]);
      } else {
        values[idx] = clamp(p.value, mins[idx], maxs[idx]);
      }
    }
  };

  function ExpressionManager(model) {
    this._model = model;
    this._current = null;
    this._cache = new Map();
  }

  ExpressionManager.prototype.load = function (file) {
    if (this._cache.has(file)) return this._cache.get(file);
    var promise = fetch(file)
      .then(function (res) {
        if (!res.ok) throw new Error("expression " + file);
        return res.json();
      })
      .then(function (json) {
        return new ExpressionEntry(json);
      });
    this._cache.set(file, promise);
    return promise;
  };

  ExpressionManager.prototype.set = function (entry) {
    this._current = entry || null;
  };

  ExpressionManager.prototype.update = function () {
    if (this._current) this._current.apply(this._model);
  };

  // ---------------------------------------------------------------- physics

  function normalizeParameterValue(
    value,
    parameterMinimum,
    parameterMaximum,
    parameterDefault,
    normalizedMinimum,
    normalizedMaximum,
    normalizedDefault,
    isInverted
  ) {
    var maxValue = Math.max(parameterMaximum, parameterMinimum);
    if (maxValue < value) value = maxValue;
    var minValue = Math.min(parameterMaximum, parameterMinimum);
    if (minValue > value) value = minValue;
    var minNormValue = Math.min(normalizedMinimum, normalizedMaximum);
    var maxNormValue = Math.max(normalizedMinimum, normalizedMaximum);
    var middleNormValue = normalizedDefault;
    var middleValue = minValue + (maxValue - minValue) / 2;
    var paramValue = value - middleValue;
    var result;
    if (paramValue > 0) {
      var nLength = maxNormValue - middleNormValue;
      var pLength = maxValue - middleValue;
      result = pLength != 0 ? paramValue * (nLength / pLength) + middleNormValue : middleNormValue;
    } else if (paramValue < 0) {
      var nLength2 = minNormValue - middleNormValue;
      var pLength2 = minValue - middleValue;
      result = pLength2 != 0 ? paramValue * (nLength2 / pLength2) + middleNormValue : middleNormValue;
    } else {
      result = middleNormValue;
    }
    return isInverted ? result : -result;
  }

  function directionToRadian(directionX, directionY, targetX, targetY) {
    var dot = directionX * targetX + directionY * targetY;
    var cross = directionX * targetY - directionY * targetX;
    return Math.atan2(cross, dot);
  }

  function CubismPhysics(model, json) {
    this._model = model;
    var meta = json.Meta || {};
    this._fps = meta.Fps || 60;
    this._gravity = { x: 0, y: -1 };
    this._wind = { x: 0, y: 0 };
    var forces = meta.EffectiveForces || {};
    if (forces.Gravity) {
      this._gravity.x = forces.Gravity.X || 0;
      this._gravity.y = forces.Gravity.Y != null ? forces.Gravity.Y : -1;
    }
    if (forces.Wind) {
      this._wind.x = forces.Wind.X || 0;
      this._wind.y = forces.Wind.Y || 0;
    }
    this._settings = [];
    var list = json.PhysicsSettings || [];
    for (var i = 0; i < list.length; i++) {
      var s = list[i];
      var setting = {
        inputs: [],
        outputs: [],
        particles: [],
        normalizationPosition: (s.Normalization || {}).Position || {
          Minimum: -10,
          Default: 0,
          Maximum: 10,
        },
        normalizationAngle: (s.Normalization || {}).Angle || {
          Minimum: -10,
          Default: 0,
          Maximum: 10,
        },
        outputsState: [],
        previousOutputs: [],
      };
      var inputs = s.Input || [];
      for (var j = 0; j < inputs.length; j++) {
        setting.inputs.push({
          sourceId: inputs[j].Source.Id,
          sourceIndex: -1,
          weight: (inputs[j].Weight != null ? inputs[j].Weight : 100) / 100,
          type: inputs[j].Type || "Angle",
          reflect: !!inputs[j].Reflect,
        });
      }
      var outputs = s.Output || [];
      for (var k = 0; k < outputs.length; k++) {
        var o = outputs[k];
        setting.outputs.push({
          destinationId: o.Destination.Id,
          destinationIndex: -1,
          vertexIndex: o.VertexIndex || 0,
          scale: o.Scale != null ? o.Scale : 1,
          weight: (o.Weight != null ? o.Weight : 100) / 100,
          type: o.Type || "Angle",
          reflect: !!o.Reflect,
          valueBelowMinimum: 0,
          valueExceededMaximum: 0,
        });
        setting.outputsState.push(0);
        setting.previousOutputs.push(0);
      }
      var vertices = s.Vertices || [];
      for (var m = 0; m < vertices.length; m++) {
        var v = vertices[m];
        var pos = v.Position || { X: 0, Y: 0 };
        setting.particles.push({
          position: { x: pos.X || 0, y: pos.Y || 0 },
          initialPosition: { x: 0, y: 0 },
          lastPosition: { x: 0, y: 0 },
          lastGravity: { x: 0, y: 1 },
          velocity: { x: 0, y: 0 },
          force: { x: 0, y: 0 },
          mobility: v.Mobility != null ? v.Mobility : 1,
          delay: v.Delay != null ? v.Delay : 1,
          acceleration: v.Acceleration != null ? v.Acceleration : 1,
          radius: v.Radius != null ? v.Radius : 0,
        });
      }
      this._settings.push(setting);
    }
    this._currentRemainTime = 0;
    this._parameterCaches = null;
    this._parameterInputCaches = null;
    this.initialize();
  }

  CubismPhysics.prototype.initialize = function () {
    for (var s = 0; s < this._settings.length; s++) {
      var setting = this._settings[s];
      var particles = setting.particles;
      particles[0].initialPosition = { x: 0, y: 0 };
      particles[0].lastPosition = { x: 0, y: 0 };
      particles[0].lastGravity = { x: 0, y: 1 };
      particles[0].velocity = { x: 0, y: 0 };
      particles[0].force = { x: 0, y: 0 };
      for (var i = 1; i < particles.length; i++) {
        particles[i].initialPosition = {
          x: particles[i - 1].initialPosition.x,
          y: particles[i - 1].initialPosition.y + particles[i].radius,
        };
        particles[i].position = {
          x: particles[i].initialPosition.x,
          y: particles[i].initialPosition.y,
        };
        particles[i].lastPosition = {
          x: particles[i].initialPosition.x,
          y: particles[i].initialPosition.y,
        };
        particles[i].lastGravity = { x: 0, y: 1 };
        particles[i].velocity = { x: 0, y: 0 };
        particles[i].force = { x: 0, y: 0 };
      }
    }
  };

  CubismPhysics.prototype.update = function (deltaTimeSeconds) {
    if (deltaTimeSeconds <= 0) return;
    var core = this._model.coreModel;
    var parameterValues = core.parameters.values;
    var parameterMaximumValues = core.parameters.maximumValues;
    var parameterMinimumValues = core.parameters.minimumValues;
    var parameterDefaultValues = core.parameters.defaultValues;
    var count = core.parameters.count;

    if (!this._parameterCaches || this._parameterCaches.length < count) {
      this._parameterCaches = new Float32Array(count);
    }
    if (
      !this._parameterInputCaches ||
      this._parameterInputCaches.length < count
    ) {
      this._parameterInputCaches = new Float32Array(count);
      for (var j = 0; j < count; j++) {
        this._parameterInputCaches[j] = parameterValues[j];
      }
    }

    this._currentRemainTime += deltaTimeSeconds;
    if (this._currentRemainTime > 0.5) this._currentRemainTime = 0;
    var physicsDeltaTime = this._fps > 0 ? 1 / this._fps : deltaTimeSeconds;

    var MAX_DELTA = 0.5;
    var AIR_RESISTANCE = 5.0;
    var MAXIMUM_WEIGHT = 100;
    var MOVEMENT_THRESHOLD = 0.001;

    while (this._currentRemainTime >= physicsDeltaTime) {
      // previous outputs <- current outputs (pendulum interpolation)
      for (var s = 0; s < this._settings.length; s++) {
        var setting = this._settings[s];
        for (var i = 0; i < setting.outputsState.length; i++) {
          setting.previousOutputs[i] = setting.outputsState[i];
        }
      }

      // interpolate the input between the last physics tick and now
      var inputWeight = physicsDeltaTime / this._currentRemainTime;
      for (var j2 = 0; j2 < count; j2++) {
        this._parameterCaches[j2] =
          this._parameterInputCaches[j2] * (1 - inputWeight) +
          parameterValues[j2] * inputWeight;
        this._parameterInputCaches[j2] = this._parameterCaches[j2];
      }

      for (var s2 = 0; s2 < this._settings.length; s2++) {
        var setting2 = this._settings[s2];
        var totalTranslation = { x: 0, y: 0 };
        var totalAngle = 0;

        // inputs
        for (var i2 = 0; i2 < setting2.inputs.length; i2++) {
          var input = setting2.inputs[i2];
          if (input.sourceIndex < 0) {
            input.sourceIndex = this._model.paramIndex(input.sourceId);
          }
          if (input.sourceIndex < 0) continue;
          var weight = input.weight; // already /100
          var normalized = normalizeParameterValue(
            this._parameterCaches[input.sourceIndex],
            parameterMinimumValues[input.sourceIndex],
            parameterMaximumValues[input.sourceIndex],
            parameterDefaultValues[input.sourceIndex],
            setting2.normalizationPosition.Minimum,
            setting2.normalizationPosition.Maximum,
            setting2.normalizationPosition.Default,
            input.reflect
          );
          if (input.type === "X") {
            totalTranslation.x += normalized * weight;
          } else if (input.type === "Y") {
            totalTranslation.y += normalized * weight;
          } else {
            var angleNorm = normalizeParameterValue(
              this._parameterCaches[input.sourceIndex],
              parameterMinimumValues[input.sourceIndex],
              parameterMaximumValues[input.sourceIndex],
              parameterDefaultValues[input.sourceIndex],
              setting2.normalizationAngle.Minimum,
              setting2.normalizationAngle.Maximum,
              setting2.normalizationAngle.Default,
              input.reflect
            );
            totalAngle += angleNorm * weight;
          }
        }

        // rotate translation by -totalAngle
        var radAngle = (-totalAngle * Math.PI) / 180;
        var tx =
          totalTranslation.x * Math.cos(radAngle) -
          totalTranslation.y * Math.sin(radAngle);
        var ty =
          totalTranslation.x * Math.sin(radAngle) +
          totalTranslation.y * Math.cos(radAngle);
        totalTranslation.x = tx;
        totalTranslation.y = ty;

        updateParticles(
          setting2.particles,
          totalTranslation,
          totalAngle,
          this._wind,
          MOVEMENT_THRESHOLD * setting2.normalizationPosition.Maximum,
          physicsDeltaTime,
          AIR_RESISTANCE
        );

        // outputs
        for (var o = 0; o < setting2.outputs.length; o++) {
          var output = setting2.outputs[o];
          var particleIndex = output.vertexIndex;
          if (output.destinationIndex < 0) {
            output.destinationIndex = this._model.paramIndex(
              output.destinationId
            );
          }
          if (
            particleIndex < 1 ||
            particleIndex >= setting2.particles.length ||
            output.destinationIndex < 0
          ) {
            continue;
          }
          var pCurrent = setting2.particles[particleIndex].position;
          var pPrev = setting2.particles[particleIndex - 1].position;
          var translationX = pCurrent.x - pPrev.x;
          var translationY = pCurrent.y - pPrev.y;

          var outputValue;
          if (output.type === "X") {
            outputValue = output.reflect ? -translationX : translationX;
          } else if (output.type === "Y") {
            outputValue = output.reflect ? -translationY : translationY;
          } else {
            // Angle: direction between the parent segment and this segment
            var parentX;
            var parentY;
            if (particleIndex >= 2) {
              var pp = setting2.particles[particleIndex - 2].position;
              parentX = pPrev.x - pp.x;
              parentY = pPrev.y - pp.y;
            } else {
              parentX = -this._gravity.x;
              parentY = -this._gravity.y;
            }
            outputValue = directionToRadian(
              parentX,
              parentY,
              translationX,
              translationY
            );
            if (output.reflect) outputValue = -outputValue;
          }

          setting2.outputsState[o] = outputValue;

          // write the output into the parameter cache
          var destinationParameterIndex = output.destinationIndex;
          var outputScale = output.scale;
          var value = outputValue * outputScale;
          var pMin = parameterMinimumValues[destinationParameterIndex];
          var pMax = parameterMaximumValues[destinationParameterIndex];
          if (value < pMin) {
            if (value < output.valueBelowMinimum) output.valueBelowMinimum = value;
            value = pMin;
          } else if (value > pMax) {
            if (value > output.valueExceededMaximum)
              output.valueExceededMaximum = value;
            value = pMax;
          }
          var w = output.weight; // already /100
          if (w >= 1) {
            this._parameterCaches[destinationParameterIndex] = value;
          } else {
            this._parameterCaches[destinationParameterIndex] =
              this._parameterCaches[destinationParameterIndex] * (1 - w) +
              value * w;
          }
        }
      }
      this._currentRemainTime -= physicsDeltaTime;
    }

    // pendulum interpolation between the last two physics ticks
    var alpha = this._currentRemainTime / physicsDeltaTime;
    for (var s3 = 0; s3 < this._settings.length; s3++) {
      var setting3 = this._settings[s3];
      for (var o2 = 0; o2 < setting3.outputs.length; o2++) {
        var output2 = setting3.outputs[o2];
        if (output2.destinationIndex < 0) continue;
        var interpolated =
          setting3.previousOutputs[o2] * (1 - alpha) +
          setting3.outputsState[o2] * alpha;
        var outputScale2 = output2.scale;
        var value2 = interpolated * outputScale2;
        var di = output2.destinationIndex;
        var pMin2 = parameterMinimumValues[di];
        var pMax2 = parameterMaximumValues[di];
        if (value2 < pMin2) value2 = pMin2;
        else if (value2 > pMax2) value2 = pMax2;
        var w2 = output2.weight;
        if (w2 >= 1) {
          parameterValues[di] = value2;
        } else {
          parameterValues[di] =
            parameterValues[di] * (1 - w2) + value2 * w2;
        }
      }
    }
  };

  function updateParticles(
    particles,
    totalTranslation,
    totalAngle,
    windDirection,
    thresholdValue,
    deltaTimeSeconds,
    airResistance
  ) {
    particles[0].position.x = totalTranslation.x;
    particles[0].position.y = totalTranslation.y;
    var totalRadian = (totalAngle * Math.PI) / 180;
    // gravity direction rotated by the total angle
    var gx = Math.sin(totalRadian);
    var gy = -Math.cos(totalRadian);
    var glen = Math.sqrt(gx * gx + gy * gy) || 1;
    gx /= glen;
    gy /= glen;

    for (var i = 1; i < particles.length; i++) {
      var p = particles[i];
      p.force.x = gx * p.acceleration + windDirection.x;
      p.force.y = gy * p.acceleration + windDirection.y;
      p.lastPosition.x = p.position.x;
      p.lastPosition.y = p.position.y;

      var delay = p.delay * deltaTimeSeconds * 30.0;
      var dirX = p.position.x - particles[i - 1].position.x;
      var dirY = p.position.y - particles[i - 1].position.y;

      var radian =
        directionToRadian(
          p.lastGravity.x,
          p.lastGravity.y,
          gx,
          gy
        ) / airResistance;
      var cosR = Math.cos(radian);
      var sinR = Math.sin(radian);
      var rx = cosR * dirX - sinR * dirY;
      var ry = sinR * dirX + cosR * dirY;

      p.position.x = particles[i - 1].position.x + rx;
      p.position.y = particles[i - 1].position.y + ry;

      var vx = p.velocity.x * delay;
      var vy = p.velocity.y * delay;
      var fx = p.force.x * delay * delay;
      var fy = p.force.y * delay * delay;
      p.position.x += vx + fx;
      p.position.y += vy + fy;

      var ndx = p.position.x - particles[i - 1].position.x;
      var ndy = p.position.y - particles[i - 1].position.y;
      var nlen = Math.sqrt(ndx * ndx + ndy * ndy) || 1;
      p.position.x = particles[i - 1].position.x + (ndx / nlen) * p.radius;
      p.position.y = particles[i - 1].position.y + (ndy / nlen) * p.radius;

      if (Math.abs(p.position.x) < thresholdValue) p.position.x = 0;

      if (delay != 0) {
        p.velocity.x = ((p.position.x - p.lastPosition.x) / delay) * p.mobility;
        p.velocity.y = ((p.position.y - p.lastPosition.y) / delay) * p.mobility;
      }
      p.force.x = 0;
      p.force.y = 0;
      p.lastGravity.x = gx;
      p.lastGravity.y = gy;
    }
  }

  // ------------------------------------------------------------- eye blink

  function EyeBlink(model, parameterIds) {
    this._model = model;
    this._ids = parameterIds || [];
    this._state = "first";
    this._time = 0;
    this._nextInterval = 2 + Math.random() * 2;
    this._closingTime = 0.1;
    this._closedTime = 0.05;
    this._openingTime = 0.15;
  }

  EyeBlink.prototype.update = function (dt) {
    this._time += dt;
    var value = 1;
    if (this._state === "first" || this._state === "interval") {
      if (this._time >= this._nextInterval) {
        this._state = "closing";
        this._time = 0;
      }
    } else if (this._state === "closing") {
      var t = this._time / this._closingTime;
      if (t >= 1) {
        value = 0;
        this._state = "closed";
        this._time = 0;
      } else {
        value = 1 - smoothstep(t);
      }
    } else if (this._state === "closed") {
      value = 0;
      if (this._time >= this._closedTime) {
        this._state = "opening";
        this._time = 0;
      }
    } else if (this._state === "opening") {
      var t2 = this._time / this._openingTime;
      if (t2 >= 1) {
        value = 1;
        this._state = "interval";
        this._time = 0;
        this._nextInterval = 2 + Math.random() * 4;
      } else {
        value = smoothstep(t2);
      }
    }
    var model = this._model;
    for (var i = 0; i < this._ids.length; i++) {
      model._setDirect(this._ids[i], value);
    }
  };

  // ---------------------------------------------------------------- breath

  function Breath() {
    // offset, amplitude, period, phase — the framework defaults
    this._params = [
      { id: "ParamAngleX", offset: 0, amplitude: 15, period: 6.5345, phase: 0.5 },
      { id: "ParamAngleY", offset: 0, amplitude: 8, period: 3.5345, phase: 0.5 },
      { id: "ParamAngleZ", offset: 0, amplitude: 10, period: 5.5345, phase: 0.5 },
      { id: "ParamBodyAngleX", offset: 0, amplitude: 4, period: 15.5345, phase: 0.5 },
      { id: "ParamBreath", offset: 0, amplitude: 0.5, period: 3.2345, phase: 0.5 },
    ];
    this._time = 0;
  }

  Breath.prototype.update = function (model, dt) {
    this._time += dt;
    for (var i = 0; i < this._params.length; i++) {
      var p = this._params[i];
    var value =
      p.offset +
      p.amplitude *
        (1 - Math.cos(((this._time / p.period + p.phase) * 2 * Math.PI))) /
        2;
    model._addDirect(p.id, value);
    }
  };

  // -------------------------------------------------------------- the model

  var BLEND_MODES = PIXI.BLEND_MODES;

  // color blend type (low byte of drawables.blendModes[i]) -> PixiJS blend
  var COLOR_BLEND_TO_PIXI = {
    0: BLEND_MODES.NORMAL, // Normal
    1: BLEND_MODES.ADD, // Add
    2: BLEND_MODES.ADD, // AddGlow
    3: BLEND_MODES.DARKEN,
    4: BLEND_MODES.MULTIPLY,
    5: BLEND_MODES.COLOR_BURN,
    6: BLEND_MODES.COLOR_BURN, // LinearBurn
    7: BLEND_MODES.LIGHTEN,
    8: BLEND_MODES.SCREEN,
    9: BLEND_MODES.COLOR_DODGE,
    10: BLEND_MODES.OVERLAY,
    11: BLEND_MODES.SOFT_LIGHT,
    12: BLEND_MODES.HARD_LIGHT,
    13: BLEND_MODES.HARD_LIGHT, // LinearLight
    14: BLEND_MODES.HUE,
    15: BLEND_MODES.COLOR,
    16: BLEND_MODES.ADD, // AddCompatible
    17: BLEND_MODES.MULTIPLY, // MultiplyCompatible
  };

  class Cubism5Model extends PIXI.Container {
    constructor(coreModel, settings, baseUrl) {
      super();

    this.coreModel = coreModel;
    this.settings = settings;
    this.baseUrl = baseUrl;

    var canvasinfo = coreModel.canvasinfo;
    this.canvasWidth = canvasinfo.CanvasWidth; // pixels
    this.canvasHeight = canvasinfo.CanvasHeight; // pixels
    this.canvasOriginX = canvasinfo.CanvasOriginX || 0;
    this.canvasOriginY = canvasinfo.CanvasOriginY || 0;
    this.pixelsPerUnit = canvasinfo.PixelsPerUnit || 1;
    // canvas size in model units. Not "width"/"height":
    // PIXI.Container defines those as accessors, so plain
    // assignments would call the setters instead of storing.
    this.canvasWidthUnits = this.canvasWidth / this.pixelsPerUnit;
    this.canvasHeightUnits =
      this.canvasHeight / this.pixelsPerUnit;

    this._paramIndexCache = new Map();
    this._partIndexCache = new Map();
    this._overrides = new Map(); // paramIndex -> {value, weight, add}

    this._motionManager = new MotionManager(this);
    this._expressionManager = new ExpressionManager(this);

    // eye blink parameters from the model3.json groups
    this._blinkIds = [];
    var groups = settings.Groups || [];
    for (var g = 0; g < groups.length; g++) {
      if (groups[g].Name === "EyeBlink" && groups[g].Target === "Parameter") {
        this._blinkIds = groups[g].Ids || [];
      }
    }
    this._eyeBlink = new EyeBlink(this, this._blinkIds);
    this._breath = new Breath();
    this._physics = null;

    // pointer focus
    this._focus = { x: 0, y: 0 };
    this._autoInteract = true;

    // inner layer: local (0, 0) is the top-center of the model
    // canvas and local y points down, in model units, so the
    // layer spans the whole canvas for any pixels-per-unit.
    this._layer = new PIXI.Container();
    this._layer.position.set(
      -this.canvasWidthUnits / 2,
      -this.canvasHeightUnits
    );
    this.addChild(this._layer);
    this._layer.sortableChildren = true;

    this._meshes = [];
    this._maskGraphics = new Map(); // drawable index -> PIXI.Graphics
    this._ready = false;
    }
  }

  Cubism5Model.MotionPriority = MotionPriority;

  Cubism5Model.ensureCoreReady = function () {
    var tries = 0;
    function probe() {
      try {
        return CORE.Version.csmGetVersion() > 0;
      } catch (e) {
        return false;
      }
    }
    return new Promise(function (resolve, reject) {
      (function attempt() {
        if (probe()) return resolve();
        if (++tries > 200) return reject(new Error("Cubism core did not start"));
        setTimeout(attempt, 10);
      })();
    });
  };

  Cubism5Model.from = function (model3Url) {
    return Cubism5Model.ensureCoreReady().then(function () {
      return fetch(model3Url).then(function (res) {
        if (!res.ok) throw new Error("model3.json " + model3Url);
        return res.json();
      });
    }).then(function (settings) {
      var baseUrl = model3Url.substring(0, model3Url.lastIndexOf("/") + 1);
      var mocUrl = baseUrl + settings.FileReferences.Moc;
      return fetch(mocUrl)
        .then(function (res) {
          if (!res.ok) throw new Error("moc3 " + mocUrl);
          return res.arrayBuffer();
        })
        .then(function (mocBytes) {
          var moc = CORE.Moc.fromArrayBuffer(mocBytes);
          if (!moc) throw new Error("could not read moc3");
          var coreModel = CORE.Model.fromMoc(moc);
          if (!coreModel) throw new Error("could not create model");
          var model = new Cubism5Model(coreModel, settings, baseUrl);
          return model._init();
        });
    });
  };

  Cubism5Model.prototype._init = function () {
    var self = this;
    var core = this.coreModel;
    var refs = (this.settings.FileReferences || {});
    var promises = [];

    // textures
    var textureFiles = refs.Textures || [];
    this._textures = [];
    for (var i = 0; i < textureFiles.length; i++) {
      (function (index, file) {
        promises.push(
          loadTexture(self.baseUrl + file).then(function (texture) {
            self._textures[index] = texture;
          })
        );
      })(i, textureFiles[i]);
    }

    // physics
    if (refs.Physics) {
      promises.push(
        fetch(this.baseUrl + refs.Physics)
          .then(function (res) {
            return res.ok ? res.json() : null;
          })
          .then(function (json) {
            if (json) self._physics = new CubismPhysics(self, json);
          })
      );
    }

    // offscreens (Cubism 5.3 render-to-texture layers) are
    // not supported; their drawables still render, just
    // without the offscreen pass.
    if (core.offscreens && core.offscreens.count > 0) {
      console.warn(
        "[cubism5] model uses offscreen layers; they will render without their offscreen pass"
      );
    }

    // The meshes need the textures, so they are built only
    // once every texture has loaded.
    return Promise.all(promises).then(function () {
      self._buildMeshes();
      self._ready = true;
      self.update(0); // one frame so the model is on screen immediately
      self._computeBounds();
      return self;
    });
  };

  // One PixiJS mesh per drawable, in draw order.
  Cubism5Model.prototype._buildMeshes = function () {
    var self = this;
    var core = this.coreModel;
    var drawables = core.drawables;
    var count = drawables.count;
    var hasMasks = false;
    for (var d = 0; d < count; d++) {
      if (drawables.maskCounts[d] > 0) {
        hasMasks = true;
        break;
      }
    }
    for (var i2 = 0; i2 < count; i2++) {
      var geometry = new PIXI.Geometry();
      var vertexCount = drawables.vertexCounts[i2];
      var positions = new Float32Array(vertexCount * 2);
      var uvs = new Float32Array(drawables.vertexUvs[i2]); // copy once
      var indices = new Uint16Array(drawables.indices[i2]);
      // PixiJS v7's mesh shader reads these attribute names.
      geometry.addAttribute("aVertexPosition", positions, 2);
      geometry.addAttribute("aTextureCoord", uvs, 2);
      geometry.addIndex(indices);
      var textureIndex = drawables.textureIndices[i2];
      var texture =
        textureIndex >= 0 &&
        textureIndex < self._textures.length &&
        self._textures[textureIndex]
          ? self._textures[textureIndex]
          : fallbackTexture();
      var material = new PIXI.MeshMaterial(texture);
      var mesh = new PIXI.Mesh(geometry, material);
      mesh.zIndex = drawables.drawOrders[i2];
      mesh.visible = false;
      this._layer.addChild(mesh);
      this._meshes.push(mesh);

      // mask graphics for drawables that clip this one
      if (hasMasks && drawables.maskCounts[i2] > 0) {
        var masks = drawables.masks[i2];
        for (var m = 0; m < masks.length; m++) {
          this._ensureMaskGraphics(masks[m]);
        }
        // PixiJS supports a single mask per object; the first mask is used
        mesh.mask = this._maskGraphics.get(masks[0]);
      }
    }
  };

  // Her actual footprint in this container's space, where
  // (0, 0) is the canvas bottom-center and y points down.
  // Her art can stick out past the canvas, so the page fits
  // her to this instead of to the canvas size.
  Cubism5Model.prototype._computeBounds = function () {
    var drawables = this.coreModel.drawables;
    var minX = Infinity;
    var maxX = -Infinity;
    var minY = Infinity;
    var maxY = -Infinity;
    for (var i = 0; i < drawables.count; i++) {
      if (!CORE.Utils.hasIsVisibleBit(drawables.dynamicFlags[i])) continue;
      var positions = drawables.vertexPositions[i];
      for (var j = 0; j < positions.length; j += 2) {
        var x = positions[j];
        var y = -positions[j + 1] - this.canvasHeightUnits / 2;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
    this.bounds = {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
    };
  };

  Cubism5Model.prototype._ensureMaskGraphics = function (drawableIndex) {
    if (this._maskGraphics.has(drawableIndex)) return;
    var graphics = new PIXI.Graphics();
    graphics.alpha = 0; // stencil masks do not need color
    this._layer.addChild(graphics);
    this._maskGraphics.set(drawableIndex, graphics);
  };

  Cubism5Model.prototype._updateMaskGraphics = function () {
    if (this._maskGraphics.size === 0) return;
    var drawables = this.coreModel.drawables;
    this._maskGraphics.forEach(function (graphics, drawableIndex) {
      var positions = drawables.vertexPositions[drawableIndex];
      var n = positions.length / 2;
      if (n === 0) return;
      graphics.clear();
      graphics.beginFill(0xffffff);
      var p = this._toLayerPoint(positions, 0);
      graphics.moveTo(p[0], p[1]);
      for (var i = 1; i < n; i++) {
        p = this._toLayerPoint(positions, i);
        graphics.lineTo(p[0], p[1]);
      }
      graphics.endFill();
    }, this);
  };

  // vertex position (model units, canvas center, y up) ->
  // inner layer coordinates (canvas top-left origin, y down)
  Cubism5Model.prototype._toLayerPoint = function (positions, vertex) {
    var x = positions[vertex * 2];
    var y = positions[vertex * 2 + 1];
    return [
      x + this.canvasWidthUnits / 2,
      this.canvasHeightUnits / 2 - y,
    ];
  };

  // ------------------------------------------------------------- parameters

  Cubism5Model.prototype.paramIndex = function (id) {
    if (this._paramIndexCache.has(id)) {
      return this._paramIndexCache.get(id);
    }
    var ids = this.coreModel.parameters.ids;
    var index = -1;
    for (var i = 0; i < ids.length; i++) {
      if (ids[i] === id) {
        index = i;
        break;
      }
    }
    this._paramIndexCache.set(id, index);
    return index;
  };

  Cubism5Model.prototype.partIndex = function (id) {
    if (this._partIndexCache.has(id)) {
      return this._partIndexCache.get(id);
    }
    var ids = this.coreModel.parts.ids;
    var index = -1;
    for (var i = 0; i < ids.length; i++) {
      if (ids[i] === id) {
        index = i;
        break;
      }
    }
    this._partIndexCache.set(id, index);
    return index;
  };

  Cubism5Model.prototype.getParameterValueById = function (id) {
    var index = this.paramIndex(id);
    if (index < 0) return 0;
    return this.coreModel.parameters.values[index];
  };

  // Internal systems (eye blink) write directly: they run inside
  // update(), after the motion base is restored, so they are
  // included in this frame's model recomputation.
  Cubism5Model.prototype._setDirect = function (id, value, weight) {
    var index = this.paramIndex(id);
    if (index < 0) return;
    if (weight == null) weight = 1;
    var core = this.coreModel;
    var values = core.parameters.values;
    var value2 = clamp(value, core.parameters.minimumValues[index], core.parameters.maximumValues[index]);
    if (weight >= 1) {
      values[index] = value2;
    } else {
      values[index] = values[index] * (1 - weight) + value2 * weight;
    }
  };

  Cubism5Model.prototype._addDirect = function (id, value, weight) {
    var index = this.paramIndex(id);
    if (index < 0) return;
    if (weight == null) weight = 1;
    var core = this.coreModel;
    var values = core.parameters.values;
    var target =
      values[index] + (weight >= 1 ? value : value * weight);
    values[index] = clamp(
      target,
      core.parameters.minimumValues[index],
      core.parameters.maximumValues[index]
    );
  };

  // Sets are deferred to the next update() so they are applied on top of the
  // automatic systems (motions, expressions, physics) but still inside the
  // same frame, right before the model recomputes its vertices.
  Cubism5Model.prototype.setParameterValueById = function (id, value, weight) {
    var index = this.paramIndex(id);
    if (index < 0) return;
    if (weight == null) weight = 1;
    this._overrides.set(index, { value: value, weight: weight, add: false });
  };

  Cubism5Model.prototype.addParameterValueById = function (id, value, weight) {
    var index = this.paramIndex(id);
    if (index < 0) return;
    if (weight == null) weight = 1;
    this._overrides.set(index, { value: value, weight: weight, add: true });
  };

  Cubism5Model.prototype._applyOverrides = function () {
    if (this._overrides.size === 0) return;
    var core = this.coreModel;
    var values = core.parameters.values;
    var minimumValues = core.parameters.minimumValues;
    var maximumValues = core.parameters.maximumValues;
    this._overrides.forEach(function (override, index) {
      var value = override.value;
      if (!override.add) {
        value = clamp(value, minimumValues[index], maximumValues[index]);
      }
      var current = values[index];
      if (override.weight >= 1) {
        values[index] = override.add ? current + value : value;
      } else {
        var target = override.add ? current + value : value;
        values[index] = current * (1 - override.weight) + target * override.weight;
      }
    });
    this._overrides.clear();
  };

  // ------------------------------------------------------------- expressions

  Cubism5Model.prototype.expression = function (name) {
    if (!name) {
      this._expressionManager.set(null);
      return Promise.resolve();
    }
    var expressions = (this.settings.FileReferences || {}).Expressions || [];
    var file = null;
    for (var i = 0; i < expressions.length; i++) {
      if (expressions[i].Name === name) {
        file = expressions[i].File;
        break;
      }
    }
    if (file == null) {
      return Promise.reject(new Error("unknown expression: " + name));
    }
    var self = this;
    return this._expressionManager
      .load(this.baseUrl + file)
      .then(function (entry) {
        self._expressionManager.set(entry);
      });
  };

  // ---------------------------------------------------------------- motions

  Cubism5Model.prototype.motion = function (group, index, priority) {
    var motions = ((this.settings.FileReferences || {}).Motions || {})[group];
    if (!motions || !motions.length) {
      return Promise.reject(new Error("no motion group: " + group));
    }
    var entry = motions[index != null ? index : 0];
    if (!entry) {
      return Promise.reject(new Error("no motion " + group + "[" + index + "]"));
    }
    var self = this;
    return this._motionManager
      .load(this.baseUrl + entry.File)
      .then(function (motionEntry) {
        self._motionManager.start(motionEntry, priority != null ? priority : MotionPriority.NORMAL);
      });
  };

  Cubism5Model.prototype.stopMotions = function () {
    this._motionManager.stop();
  };

  // ------------------------------------------------------------------ focus

  Cubism5Model.prototype.setFocus = function (x, y) {
    this._focus.x = clamp(x, -1, 1);
    this._focus.y = clamp(y, -1, 1);
  };

  Cubism5Model.prototype._updateFocus = function () {
    if (!this._autoInteract) return;
    var x = this._focus.x;
    var y = this._focus.y;
    this._addDirect("ParamEyeBallX", x);
    this._addDirect("ParamEyeBallY", y);
    this._addDirect("ParamAngleX", x * 30);
    this._addDirect("ParamAngleY", y * 30);
    this._addDirect("ParamAngleZ", x * y * -30);
    this._addDirect("ParamBodyAngleX", x * 10);
  };

  // -------------------------------------------------------------- the frame

  Cubism5Model.prototype.update = function (deltaTimeMilliseconds) {
    var dt = deltaTimeMilliseconds / 1000;
    var core = this.coreModel;

    // 1. restore the parameter base (the saved motion state)
    var savedParameters = this._savedParameters;
    if (!savedParameters) {
      savedParameters = this._savedParameters = new Float32Array(
        core.parameters.values
      );
    } else if (savedParameters.length !== core.parameters.count) {
      savedParameters = this._savedParameters = new Float32Array(
        core.parameters.values
      );
    } else {
      core.parameters.values.set(savedParameters);
    }

    // 2. motions
    var motionActive = this._motionManager.update(dt);

    // 3. save the motion result as the new base
    this._savedParameters.set(core.parameters.values);

    // 4. eye blink (runs on top of any motion, like the framework)
    this._eyeBlink.update(dt);

    // 5. expressions
    this._expressionManager.update();

    // 6. pointer focus
    this._updateFocus();

    // 7. breath
    this._breath.update(this, dt);

    // 8. physics
    if (this._physics) this._physics.update(dt);

    // 9. external parameter sets (lip sync, thinking tilt, ...)
    this._applyOverrides();

    // 10. recompute the model
    core.update();

    // 11. push the results into the PixiJS meshes
    this._syncDrawables();
  };

  Cubism5Model.prototype._syncDrawables = function () {
    var core = this.coreModel;
    var drawables = core.drawables;
    // The core's vertex positions are already in model units,
    // centered on the canvas, with y pointing up. The inner
    // layer's space has its origin at the canvas top-left with
    // y pointing down, so the conversion is a shift and a flip.
    var halfW = this.canvasWidthUnits / 2;
    var halfH = this.canvasHeightUnits / 2;
    var multiplyColors = drawables.multiplyColors;

    // blend mode layout detection: the core packs the color blend (low byte)
    // and the alpha blend (high byte) into one int per drawable
    for (var i = 0; i < drawables.count; i++) {
      var mesh = this._meshes[i];
      var dynamicFlags = drawables.dynamicFlags[i];
      var visible = CORE.Utils.hasIsVisibleBit(dynamicFlags);
      mesh.visible = visible;
      if (!visible) continue;

      // vertex positions
      var positions = drawables.vertexPositions[i];
      var attr = mesh.geometry.getBuffer("aVertexPosition");
      var arr = attr.data;
      for (var v = 0; v < positions.length; v += 2) {
        arr[v] = positions[v] + halfW;
        arr[v + 1] = halfH - positions[v + 1];
      }
      attr.update();

      mesh.alpha = drawables.opacities[i];
      mesh.zIndex = core.renderOrders[i];

      // blend mode
      var packed = drawables.blendModes[i];
      var colorBlend = packed & 0xff;
      if (colorBlend > 17) {
        // unexpected layout: fall back to the classic alpha blend flags
        var constantFlags = drawables.constantFlags[i];
        if (CORE.Utils.hasBlendAdditiveBit(constantFlags)) {
          mesh.blendMode = BLEND_MODES.ADD;
        } else if (CORE.Utils.hasBlendMultiplicativeBit(constantFlags)) {
          mesh.blendMode = BLEND_MODES.MULTIPLY;
        } else {
          mesh.blendMode = BLEND_MODES.NORMAL;
        }
      } else {
        mesh.blendMode = COLOR_BLEND_TO_PIXI[colorBlend] || BLEND_MODES.NORMAL;
      }

      // multiply color as tint (skip when it is white)
      var mr = multiplyColors[i * 4];
      var mg = multiplyColors[i * 4 + 1];
      var mb = multiplyColors[i * 4 + 2];
      if (mr != null && (mr !== 1 || mg !== 1 || mb !== 1)) {
        var r = clamp(Math.round(mr * 255), 0, 255);
        var g = clamp(Math.round(mg * 255), 0, 255);
        var b = clamp(Math.round(mb * 255), 0, 255);
        mesh.tint = (r << 16) | (g << 8) | b;
      } else if (mesh.tint !== 0xffffff) {
        mesh.tint = 0xffffff;
      }
    }

    this._updateMaskGraphics();
  };

  // ------------------------------------------------------------------- load

  var _fallbackTexture = null;
  function fallbackTexture() {
    if (_fallbackTexture) return _fallbackTexture;
    var canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    var context = canvas.getContext("2d");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, 1, 1);
    _fallbackTexture = PIXI.Texture.from(canvas);
    return _fallbackTexture;
  }

  function loadTexture(url) {
    return new Promise(function (resolve) {
      var texture = PIXI.Texture.from(url);
      if (texture.baseTexture.valid) return resolve(texture);
      var baseTexture = texture.baseTexture;
      var done = false;
      function finish() {
        if (done) return;
        done = true;
        baseTexture.off("loaded", finish);
        baseTexture.off("error", finish);
        resolve(texture);
      }
      baseTexture.on("loaded", finish);
      baseTexture.on("error", finish);
    });
  }

  global.Cubism5Model = Cubism5Model;
})(window);
