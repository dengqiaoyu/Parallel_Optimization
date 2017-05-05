if (self.CavalryLogger) { CavalryLogger.start_js(["mkxfq"]); }

__d("PaymentItemTypeFields",[],(function a(b,c,d,e,f,g){f.exports={NONE:"none",ADS_INVOICE:"ads_invoice",DONATIONS:"donations",OCULUS_CV1:"oculus_cv1",OCULUS_LAUNCH_V1:"oculus_launch_v1",OCULUS_LAUNCH_V2:"oculus_launch_v2",OZONE:"ozone",OPEN_GRAPH_PRODUCT:"open_graph_product",MESSENGER_COMMERCE:"messenger_commerce",P2P_TRANSFER:"p2p_transfer",DUMMY_FIRST_PARTY:"dummy_first_party",DUMMY_THIRD_PARTY:"dummy_third_party",GIFTS:"gifts",BILL:"bill",AIRMAIL:"airmail",EVENT_TICKETING:"event_ticketing",PAYMENT_LITE:"payment_lite",MESSENGER_API_FEE:"messenger_api_fee",WORKPLACE:"workplace",NMOR_PAGES_COMMERCE:"nmor_pages_commerce"};}),null);
__d('CoverPromotionHandler',['AsyncDialog','AsyncRequest','BanzaiODS'],(function a(b,c,d,e,f,g){var h={handleChooseClick:function i(j,k){j.subscribe('itemclick',function(l,m){c('BanzaiODS').bumpEntityKey('cover_photo_upsell','choose');var n=new (c('AsyncRequest'))(m.item.getValue()).setRelativeTo(k);c('AsyncDialog').send(n,function(o){});});}};f.exports=h;}),null);
__d('FileFormDisableInFlight',['Form'],(function a(b,c,d,e,f,g){function h(i){'use strict';this._form=i;}h.prototype.enable=function(){'use strict';this._subscription=this._form.subscribe(['submit','initial'],this._handle.bind(this));};h.prototype.disable=function(){'use strict';this._subscription.unsubscribe();this._subscription=null;};h.prototype._handle=function(i){'use strict';if(i==='submit'){setTimeout(c('Form').setDisabled.bind(null,this._form.getRoot(),true),0);}else c('Form').setDisabled(this._form.getRoot(),false);};f.exports=h;}),null);
__d('FileFormResetOnSubmit',['DOMQuery','Event','emptyFunction'],(function a(b,c,d,e,f,g){function h(j,k){var l=c('Event').listen(j,'change',c('emptyFunction').thatReturnsFalse,c('Event').Priority.URGENT);try{k();}catch(m){throw m;}finally{l.remove();}}function i(j){'use strict';this._form=j;}i.prototype.enable=function(){'use strict';var j=this._reset.bind(this);this._subscription=this._form.subscribe('submit',function(){setTimeout(j,0);});};i.prototype.disable=function(){'use strict';this._subscription.unsubscribe();this._subscription=null;};i.prototype._reset=function(){'use strict';var j=this._form.getRoot();h(j,function(){var k=c('DOMQuery').scry(j,'input[type="file"]');k.forEach(function(l){l.value='';});});};f.exports=i;}),null);
__d('UntrustedLink',['DOM','Event','URI','UserAgent_DEPRECATED','LinkshimHandlerConfig'],(function a(b,c,d,e,f,g){function h(i,j,k){this.dom=i;this.url=i.href;this.func_get_params=k||function(){return {};};c('Event').listen(this.dom,'click',this.onclick.bind(this));c('Event').listen(this.dom,'mousedown',this.onmousedown.bind(this));c('Event').listen(this.dom,'mouseup',this.onmouseup.bind(this));c('Event').listen(this.dom,'mouseout',this.onmouseout.bind(this));this.onmousedown(c('Event').$E(j));}h.bootstrap=function(i,j,k){if(i.__untrusted)return;i.__untrusted=true;new h(i,j,k);};h.prototype.getRewrittenURI=function(){var i=babelHelpers['extends']({u:this.url,h:c('LinkshimHandlerConfig').untrusted_link_default_hash},this.func_get_params(this.dom)),j=new (c('URI'))('/l.php').setDomain(c('LinkshimHandlerConfig').linkshim_host);if(new (c('URI'))(this.url).getProtocol()=='https'){return j.setQueryData(i).setProtocol('https');}else return j.setQueryData(i).setProtocol('http');};h.prototype.onclick=function(){setTimeout(function(){this.setHref(this.url);}.bind(this),100);this.setHref(this.getRewrittenURI());};h.prototype.onmousedown=function(i){if(i.button==2)this.setHref(this.getRewrittenURI());};h.prototype.onmouseup=function(){this.setHref(this.getRewrittenURI());};h.prototype.onmouseout=function(){this.setHref(this.url);};h.prototype.setHref=function(i){if(c('UserAgent_DEPRECATED').ie()<9){var j=c('DOM').create('span');c('DOM').appendContent(this.dom,j);this.dom.href=i;c('DOM').remove(j);}else this.dom.href=i;};f.exports=h;}),null);
__d('FormSubmitOnChange',['Event','submitForm'],(function a(b,c,d,e,f,g){function h(i){'use strict';this._form=i;}h.prototype.enable=function(){'use strict';this._listener=c('Event').listen(this._form.getRoot(),'change',this._submit.bind(this));};h.prototype.disable=function(){'use strict';this._listener.remove();this._listener=null;};h.prototype._submit=function(){'use strict';c('submitForm')(this._form.getRoot());};Object.assign(h.prototype,{_listener:null});f.exports=h;}),null);
__d('ResizeListener',['EventListener','SubscriptionList','requestAnimationFrame'],(function a(b,c,d,e,f,g){'use strict';var h=void 0,i=false,j=new (c('SubscriptionList'))(function(){h=c('EventListener').listen(window,'resize',k);},function(){h.remove();});function k(){if(!i){i=true;c('requestAnimationFrame')(function(){j.fireCallbacks();i=false;});}}f.exports=j;}),null);
__d('GraphQLSubscribeModes',[],(function a(b,c,d,e,f,g){'use strict';var h={SUBSCRIBE:'subscribe',UNSUBSCRIBE:'unsubscribe'};f.exports=h;}),null);
__d('GraphQLMutationDataHandler',['invariant'],(function a(b,c,d,e,f,g,h){'use strict';var i={getMutationType:function j(k){!!k&&Object.keys(k).length===1||h(0);return Object.keys(k)[0];}};f.exports=i;}),null);
__d('GraphQLSubscriptionTracker',[],(function a(b,c,d,e,f,g){'use strict';var h={},i={trackSubscription:function j(k,l){h[l]=k;},getSubscription:function j(k){return h[k];},removeSubscription:function j(k){delete h[k];}};f.exports=i;}),null);
__d('RelayGraphQLSubscriptionNetworkLayer',['invariant'],(function a(b,c,d,e,f,g,h){'use strict';var i=void 0,j={injectGraphQLSubscriber:function k(l){i=l;},sendSubscribeRequest:function k(l){i||h(0);i.sendSubscribeRequest(l);}};f.exports=j;}),null);
__d('RQLSubscription',['invariant','GraphQLMutationDataHandler','GraphQLSubscribeModes','GraphQLSubscriptionTracker','RelayGraphQLSubscriptionNetworkLayer'],(function a(b,c,d,e,f,g,h){'use strict';var i={sendSubscribe:function j(k,l,m,n,o){c('RelayGraphQLSubscriptionNetworkLayer').sendSubscribeRequest({mode:c('GraphQLSubscribeModes').SUBSCRIBE,subscription:k,subscriptionQueryID:n,params:l,clientSubscriptionID:m,topic:o});},sendUnsubscribe:function j(k,l,m,n,o){c('RelayGraphQLSubscriptionNetworkLayer').sendSubscribeRequest({mode:c('GraphQLSubscribeModes').UNSUBSCRIBE,subscription:k,subscriptionQueryID:n,params:l,clientSubscriptionID:m,topic:o});},handleSubscriptionPayload:function j(k){var l=c('GraphQLMutationDataHandler').getMutationType(k);k[l]||h(0);var m='client_subscription_id' in k[l];if(!m)return;var n=k[l].client_subscription_id;n||h(0);var o=c('GraphQLSubscriptionTracker').getSubscription(n);if(o)o.handlePayload(k);}};f.exports=i;}),null);
__d('RelayGraphQLSubscriber',['Promise','ChannelManager','GraphQLSubscribeModes','GraphQLSubscriptionsLoggerEvent','GraphQLSubscriptionsMechanism','GraphQLSubscriptionsTypedLogger','RelayMetaRoute','RelayQuery','RQLSubscription','RTISubscriptionManager','URI','RelayRTIGraphQLSubscriberUtils','flattenRelayQuery','nullthrows','printRelayQuery'],(function a(b,c,d,e,f,g){'use strict';var h=c('RelayRTIGraphQLSubscriberUtils').getTopicAndTransformContext,i={},j=155160167952447;c('ChannelManager').startChannelManager&&c('ChannelManager').startChannelManager();function k(l){return {sendSubscribeRequest:function m(n){var o=void 0,p=this._prepareData(n);if(n.mode===c('GraphQLSubscribeModes').SUBSCRIBE){o=h(l,p.queryText,p.queryID,p.queryParams,p.topic).then(function(q){var r=q.topic,s=q.transformContext;p.topic=r;s.appID=j;return this._subscribe(p,s);}.bind(this));}else if(n.mode===c('GraphQLSubscribeModes').UNSUBSCRIBE){o=this._unsubscribe(p);}else o=c('Promise').reject('invalid mode');o['catch'](function(q){return false;});},_prepareData:function m(n){var o={input:JSON.stringify(babelHelpers['extends']({},n.params,{client_subscription_id:n.clientSubscriptionID}))},p=c('RelayQuery').Subscription.create(n.subscription,c('RelayMetaRoute').get('$GraphQLSubscriber'),o),q=c('printRelayQuery')(c('nullthrows')(c('flattenRelayQuery')(p))).text;return {queryText:q,queryID:n.subscriptionQueryID!=null?Number(n.subscriptionQueryID):null,queryParams:o,topic:n.topic!=null?String(n.topic):null,tokenKey:JSON.stringify({params:n.params,text:q}),logger:new (c('GraphQLSubscriptionsTypedLogger'))().setSubscriptionCall(p.getCall().name).setQueryParams(o).setMechanism(c('GraphQLSubscriptionsMechanism').SKYWALKER)};},_subscribe:function m(n,o){var p=n.topic,q=n.tokenKey,r=n.logger;return new (c('Promise'))(function(s){i[q]=c('RTISubscriptionManager').subscribe(c('nullthrows')(p),function(t){var u=JSON.parse(t.payload);r.setEvent(c('GraphQLSubscriptionsLoggerEvent').RECEIVEPAYLOAD).log();c('RQLSubscription').handleSubscriptionPayload(u);},{transformContext:JSON.stringify(o)});r.setEvent(c('GraphQLSubscriptionsLoggerEvent').CLIENT_SUBSCRIBE).log();s();});},_unsubscribe:function m(n){var o=n.tokenKey,p=n.logger;return new (c('Promise'))(function(q){i[o].unsubscribe();delete i[o];p.setEvent(c('GraphQLSubscriptionsLoggerEvent').CLIENT_UNSUBSCRIBE).log();q();});}};}f.exports=k;}),null);
__d('GraphQLSubscription',['invariant','ClientIDs','GraphQLMutationDataHandler','GraphQLMutatorHub','GraphQLSubscriptionTracker','RQLSubscription','RelayMetaRoute','RelayQuery','RelayStore','nullthrows'],(function a(b,c,d,e,f,g,h){'use strict';function i(j,k){this.clientSubscriptionID=c('ClientIDs').getNewClientID();this.$GraphQLSubscription1=this.constructor.getSubscriptionQuery();this.$GraphQLSubscription2=this.constructor.getSubscriptionQueryID();this.$GraphQLSubscription3=this.getQueryParams(j);this.$GraphQLSubscription4=k;this.$GraphQLSubscription5=this.getSubscriptionTopic(this.$GraphQLSubscription3);this.$GraphQLSubscription1&&this.$GraphQLSubscription3||h(0);c('RQLSubscription').sendSubscribe(this.$GraphQLSubscription1,this.$GraphQLSubscription3,this.clientSubscriptionID,this.$GraphQLSubscription2,this.$GraphQLSubscription5);c('GraphQLSubscriptionTracker').trackSubscription(this,this.clientSubscriptionID);this.$GraphQLSubscription6=true;}i.prototype.dispose=function(){this.$GraphQLSubscription6||h(0);c('RQLSubscription').sendUnsubscribe(this.$GraphQLSubscription1,this.$GraphQLSubscription3,this.clientSubscriptionID,this.$GraphQLSubscription2,this.$GraphQLSubscription5);c('GraphQLSubscriptionTracker').removeSubscription(this.clientSubscriptionID);this.$GraphQLSubscription6=false;};i.getSubscriptionQuery=function(){throw new Error('Must implement `getSubscriptionQuery.`');};i.getSubscriptionQueryID=function(){return null;};i.prototype.getSubscriptionTopic=function(){if(!this.constructor.getInputFieldsThatDetermineTopic)return null;return ['gqls',this.$GraphQLSubscription1.calls[0].name,this.constructor.getInputFieldsThatDetermineTopic().sort().map(function(j){return String(j)+'_'+String(c('nullthrows')(this.$GraphQLSubscription3[j]));}.bind(this)).join('_')].join('/');};i.prototype.getQueryParams=function(j){throw new Error('Must implement `getQueryParams.`');};i.prototype.handlePayload=function(j){this.$GraphQLSubscription6||h(0);var k=c('GraphQLMutationDataHandler').getMutationType(j),l=j[k];c('RelayStore').getStoreData().handleUpdatePayload(c('RelayQuery').Subscription.create(this.$GraphQLSubscription1,c('RelayMetaRoute').get('$GraphQLSubscription'),{input:this.$GraphQLSubscription3}),l,{configs:c('GraphQLMutatorHub').getConfigs(k)||[],isOptimisticUpdate:false});this.$GraphQLSubscription4&&this.$GraphQLSubscription4(j);};f.exports=i;}),null);
__d("XDliteGraphQLSkywalkerController",["XController"],(function a(b,c,d,e,f,g){f.exports=c("XController").create("\/dlite\/skywalker_topic\/",{});}),null);
__d("XPagesBanUserDataController",["XController"],(function a(b,c,d,e,f,g){f.exports=c("XController").create("\/pages\/admin\/ban_user\/",{page_id:{type:"FBID",required:true},user_ids:{type:"FBIDVector",required:true}});}),null);